"""Memory-bounded streaming runtime for local ball re-tracking.

The legacy runtime retained every decoded native frame for the life of a match. This runtime
keeps crops as uint8 until GPU transfer, overlaps a bounded producer with inference, and writes
candidate batches incrementally. ``--legacy-buffered`` remains available for exact comparisons.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from cv.pipeline import ball_local_refine as baseline
from cv.pipeline import ball_trt, cuda_blocking_sync
from cv.pipeline import resolution as res
from cv.pipeline.ball_local_refine import RecoveryRegion


def _contexts(temporal_ensemble: bool) -> tuple[tuple[tuple[int, ...], int], ...]:
    if temporal_ensemble:
        return (
            ((-2, -1, 0), 2),
            ((-1, 0, 1), 1),
            ((0, 1, 2), 0),
        )
    return (((-1, 0, 1), 1),)


def resident_bytes() -> int:
    """Return current process RSS without adding a monitoring dependency."""
    fields = Path("/proc/self/statm").read_text().split()
    return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")


@dataclass
class ResidentMemoryBudget:
    limit_bytes: int
    peak_bytes: int = 0

    @classmethod
    def from_gb(cls, limit_gb: float) -> ResidentMemoryBudget:
        return cls(int(limit_gb * 1_000_000_000))

    def sample(self) -> int:
        current = resident_bytes()
        self.peak_bytes = max(self.peak_bytes, current)
        if current > self.limit_bytes:
            raise MemoryError(
                "streaming crop refiner exceeded --max-resident-gb: "
                f"rss={current / 1e9:.3f} GB limit={self.limit_bytes / 1e9:.3f} GB"
            )
        return current

    def wait_for_capacity(self, reserve_bytes: int, stop: threading.Event) -> None:
        if reserve_bytes >= self.limit_bytes:
            raise MemoryError(
                "one prepared crop batch cannot fit under --max-resident-gb: "
                f"reserve={reserve_bytes / 1e9:.3f} GB "
                f"limit={self.limit_bytes / 1e9:.3f} GB"
            )
        wait_started = time.monotonic()
        while not stop.is_set():
            if self.sample() + reserve_bytes <= self.limit_bytes:
                return
            if time.monotonic() - wait_started >= 5.0:
                current = self.sample()
                raise MemoryError(
                    "streaming crop refiner cannot make progress under --max-resident-gb: "
                    f"rss={current / 1e9:.3f} GB reserve={reserve_bytes / 1e9:.3f} GB "
                    f"limit={self.limit_bytes / 1e9:.3f} GB"
                )
            time.sleep(0.05)


def _bounded_frames(
    region: RecoveryRegion,
    contexts: tuple[tuple[tuple[int, ...], int], ...],
    frame_count: int,
) -> tuple[int, ...]:
    return tuple(
        max(1, min(region.frame + offset, frame_count))
        for offsets, _ in contexts
        for offset in offsets
    )


def _read_jpeg(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise OSError(f"cannot read {path}")
    return image


def _crop_uint8(
    frame: np.ndarray,
    region: RecoveryRegion,
    artifact_size: res.FrameSize,
) -> np.ndarray:
    """Apply the legacy crop geometry but defer float normalization to the GPU."""
    image_size = res.FrameSize(frame.shape[1], frame.shape[0])
    center = res.scale_points(
        np.array([region.center_x, region.center_y]), artifact_size, image_size
    )
    crop_size = res.scale_points(np.array([region.width, region.height]), artifact_size, image_size)
    transform = np.array(
        [
            [crop_size[0] / baseline.INPUT_WH[0], 0.0, center[0] - crop_size[0] / 2.0],
            [0.0, crop_size[1] / baseline.INPUT_WH[1], center[1] - crop_size[1] / 2.0],
        ],
        dtype=np.float32,
    )
    crop = cv2.warpAffine(
        frame,
        transform,
        baseline.INPUT_WH,
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)


_DECODED_LAST: dict[int, dict] = {}


def _prepare_batch_uint8(
    chunk: list[RecoveryRegion],
    contexts: tuple[tuple[tuple[int, ...], int], ...],
    frame_counts: dict[str, int],
    frames_dir: Path,
    artifact_size: res.FrameSize,
    executor: ThreadPoolExecutor,
):
    import torch

    keys = sorted(
        {
            (region.clip, frame)
            for region in chunk
            for frame in _bounded_frames(region, contexts, frame_counts[region.clip])
        }
    )
    # Consecutive batches of one clip share context frames: reuse the previous batch's
    # decoded pictures (same bytes, same decoder) instead of reading them again.
    previous = _DECODED_LAST.get(id(executor), {})
    missing = [key for key in keys if key not in previous]
    paths = [frames_dir / clip / f"f_{frame:04d}.jpg" for clip, frame in missing]
    decoded = {key: previous[key] for key in keys if key in previous}
    decoded.update(zip(missing, executor.map(_read_jpeg, paths), strict=True))
    _DECODED_LAST[id(executor)] = decoded

    def prepare_region(region: RecoveryRegion) -> list[np.ndarray]:
        crop_cache: dict[int, np.ndarray] = {}
        samples = []
        for offsets, _ in contexts:
            context = []
            for frame_offset in offsets:
                frame = max(
                    1,
                    min(region.frame + frame_offset, frame_counts[region.clip]),
                )
                if frame not in crop_cache:
                    crop_cache[frame] = _crop_uint8(
                        decoded[region.clip, frame], region, artifact_size
                    )
                context.append(crop_cache[frame])
            samples.append(np.concatenate(context, axis=0))
        return samples

    nested = executor.map(prepare_region, chunk)
    samples = np.stack([sample for region_samples in nested for sample in region_samples])
    # Pin only the compact uint8 tensor. Float conversion and normalization happen after the
    # asynchronous host-to-device copy, reducing each queued batch by exactly four times.
    return torch.from_numpy(samples).pin_memory()


def _prepare_batch(
    chunk: list[RecoveryRegion],
    contexts: tuple[tuple[tuple[int, ...], int], ...],
    frame_counts: dict[str, int],
    read_frame,
    artifact_size: res.FrameSize,
) -> np.ndarray:
    """Legacy-shape helper retained for focused crop-reuse tests."""
    samples = []
    for region in chunk:
        crop_cache = {}
        for offsets, _ in contexts:
            context = []
            for frame_offset in offsets:
                frame = max(
                    1,
                    min(region.frame + frame_offset, frame_counts[region.clip]),
                )
                if frame not in crop_cache:
                    crop_cache[frame] = baseline._crop_tensor(
                        read_frame(region.clip, frame), region, artifact_size
                    )
                context.append(crop_cache[frame])
            samples.append(np.concatenate(context, axis=0))
    return np.stack(samples)


def _batch_reserve_bytes(
    chunk: list[RecoveryRegion],
    contexts: tuple[tuple[tuple[int, ...], int], ...],
    frame_counts: dict[str, int],
) -> int:
    unique_frames = {
        (region.clip, frame)
        for region in chunk
        for frame in _bounded_frames(region, contexts, frame_counts[region.clip])
    }
    decoded_native = len(unique_frames) * 1920 * 1080 * 3
    samples = len(chunk) * len(contexts) * 9 * baseline.INPUT_WH[0] * baseline.INPUT_WH[1]
    # numpy sample plus pinned copy, region crop cache, OpenCV scratch, and conservative slack.
    return decoded_native + samples * 3 + 256 * 1024 * 1024


def _rows_from_outputs(
    chunk: list[RecoveryRegion],
    outputs: np.ndarray,
    contexts: tuple[tuple[tuple[int, ...], int], ...],
    k_best: int,
    peak_threshold: float,
    nms_radius: int,
    temporal_ensemble: bool,
    subpixel: str,
    artifact_size: res.FrameSize,
    native_support_audit: dict | None = None,
) -> list[dict]:
    if outputs.ndim == 4 and outputs.shape[0] == len(chunk):
        # Already reduced on the GPU to each context's target channel (regions, contexts, H, W):
        # the same values ensemble_target_heatmaps stacks, averaged the same way.
        heatmaps = np.mean(outputs, axis=1)
    else:
        outputs = outputs.reshape(len(chunk), len(contexts), *outputs.shape[1:])
        heatmaps = baseline.ensemble_target_heatmaps(
            outputs, tuple(channel for _, channel in contexts)
        )
    rows = []
    for region, heatmap in zip(chunk, heatmaps, strict=True):
        for rank, (x, y, score) in enumerate(
            baseline.supported_peaks(
                heatmap,
                region,
                artifact_size,
                k_best,
                peak_threshold,
                nms_radius,
                subpixel,
                native_support_audit,
            )
        ):
            artifact_x, artifact_y = baseline.heatmap_to_artifact(
                x,
                y,
                heatmap.shape[1],
                heatmap.shape[0],
                region,
            )
            rows.append(
                {
                    "clip": region.clip,
                    "frame": f"f_{region.frame:04d}.jpg",
                    "x": artifact_x,
                    "y": artifact_y,
                    "score": score,
                    "on_court": True,
                    "rank": rank,
                    "crop_provenance": region.provenance,
                    "crop_center_x": region.center_x,
                    "crop_center_y": region.center_y,
                    "crop_width": region.width,
                    "crop_height": region.height,
                    "temporal_context": ("past_center_future" if temporal_ensemble else "centered"),
                }
            )
    return rows


def infer_region_batches(
    model,
    frames_dir: Path,
    regions: list[RecoveryRegion],
    device: int,
    artifact_size: res.FrameSize,
    batch_size: int = 32,
    k_best: int = 5,
    peak_threshold: float = 0.05,
    nms_radius: int = 4,
    temporal_ensemble: bool = True,
    subpixel: str = "centroid",
    max_resident_gb: float = 32.0,
    prefetch_batches: int = 2,
    decode_workers: int = 8,
    native_support_audit: dict | None = None,
) -> Iterator[list[dict]]:
    """Yield result batches in legacy row order under a bounded host-memory contract."""
    for rows in infer_region_batches_multi(
        [model],
        frames_dir,
        regions,
        device,
        artifact_size,
        batch_size,
        k_best,
        peak_threshold,
        nms_radius,
        temporal_ensemble,
        subpixel,
        max_resident_gb,
        prefetch_batches,
        decode_workers,
        [native_support_audit],
    ):
        yield rows[0]


def infer_region_batches_multi(
    models: list,
    frames_dir: Path,
    regions: list[RecoveryRegion],
    device: int,
    artifact_size: res.FrameSize,
    batch_size: int = 32,
    k_best: int = 5,
    peak_threshold: float = 0.05,
    nms_radius: int = 4,
    temporal_ensemble: bool = True,
    subpixel: str = "centroid",
    max_resident_gb: float = 32.0,
    prefetch_batches: int = 2,
    decode_workers: int = 8,
    native_support_audits: list[dict | None] | None = None,
) -> Iterator[list[list[dict]]]:
    """Per batch, each model's rows: several models of one crop pass (same regions, e.g. WASB
    and TrackNetV2) share one decode and crop of every batch; each model's rows are exactly
    those :func:`infer_region_batches` yields for it alone."""
    import torch

    audits = native_support_audits or [None] * len(models)

    contexts = _contexts(temporal_ensemble)
    regions_per_batch = max(1, batch_size // len(contexts))
    chunks = [
        regions[offset : offset + regions_per_batch]
        for offset in range(0, len(regions), regions_per_batch)
    ]
    frame_counts = {
        clip: sum(1 for _ in (frames_dir / clip).glob("f_*.jpg"))
        for clip in {region.clip for region in regions}
    }
    budget = ResidentMemoryBudget.from_gb(max_resident_gb)
    budget.sample()
    prepared: queue.Queue = queue.Queue(maxsize=prefetch_batches)
    slots = threading.BoundedSemaphore(prefetch_batches)
    stop = threading.Event()
    sentinel = object()

    def put(item) -> bool:
        while not stop.is_set():
            try:
                prepared.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        try:
            with ThreadPoolExecutor(
                max_workers=decode_workers, thread_name_prefix="ball-crop-jpeg"
            ) as executor:
                try:
                    for chunk in chunks:
                        while not stop.is_set() and not slots.acquire(timeout=0.1):
                            pass
                        if stop.is_set():
                            return
                        reserve = _batch_reserve_bytes(chunk, contexts, frame_counts)
                        budget.wait_for_capacity(reserve, stop)
                        batch = _prepare_batch_uint8(
                            chunk, contexts, frame_counts, frames_dir, artifact_size, executor
                        )
                        budget.sample()
                        if not put((chunk, batch)):
                            return
                finally:
                    _DECODED_LAST.pop(id(executor), None)
            put(sentinel)
        except BaseException as error:
            put(error)

    producer = threading.Thread(target=produce, name="ball-crop-producer", daemon=True)
    producer.start()
    # TENNIS_BALL_REFINE_FP16=1: half-precision model (OUTPUT-CHANGING, near-exact; measured
    # in cv/experiments/upstream_dive). Normalisation stays float32; only the forward is fp16.
    # TensorRT fp16 engines (TENNIS_BALL_ENGINE=trt_fp16, wrapped by the caller) take the same
    # half-precision inputs.
    half = os.environ.get("TENNIS_BALL_REFINE_FP16", "0") == "1" or any(
        ball_trt.is_half(model) for model in models
    )
    if half:
        models = [
            model if ball_trt.is_half(model) else model.half().to(memory_format=torch.channels_last)
            for model in models
        ]
    means = torch.as_tensor(baseline.MEAN, device=f"cuda:{device}").repeat(3).view(1, 9, 1, 1)
    stds = torch.as_tensor(baseline.STD, device=f"cuda:{device}").repeat(3).view(1, 9, 1, 1)
    in_flight: deque = deque()
    # One single-thread executor per model: a model's batches run in order (its native-support
    # audit is updated by one thread at a time) while different models overlap.
    pools = [
        ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"ball-crop-model-{index}")
        for index in range(len(models))
    ]
    try:
        while True:
            item = prepared.get()
            if item is sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            chunk, samples = item
            slots.release()
            inputs = samples.to(f"cuda:{device}", non_blocking=True).float()
            inputs.div_(255.0).sub_(means).div_(stds)
            if half:
                inputs = inputs.to(torch.float16, memory_format=torch.channels_last)

            def run_model(index: int, chunk=chunk, inputs=inputs) -> list[dict]:
                with torch.inference_mode():
                    raw = models[index](inputs)[0]
                    raw = raw.view(len(chunk), len(contexts), *raw.shape[1:])
                    # Only each context's target channel is used: pick it on the GPU and copy
                    # a third of the heatmap bytes back (sigmoid is elementwise, values equal).
                    picked = torch.stack(
                        [raw[:, i, channel] for i, (_, channel) in enumerate(contexts)], dim=1
                    )
                    outputs = picked.float().sigmoid().cpu().numpy()
                budget.sample()
                return _rows_from_outputs(
                    chunk,
                    outputs,
                    contexts,
                    k_best,
                    peak_threshold,
                    nms_radius,
                    temporal_ensemble,
                    subpixel,
                    artifact_size,
                    audits[index],
                )

            # Models run in their own threads and one batch ahead of the yield, so one model's
            # peak finding overlaps the other's forward; rows are still yielded in batch order.
            in_flight.append(
                [pools[index].submit(run_model, index) for index in range(len(models))]
            )
            del samples, inputs
            if len(in_flight) > 1:
                yield [future.result() for future in in_flight.popleft()]
        while in_flight:
            yield [future.result() for future in in_flight.popleft()]
    finally:
        stop.set()
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=True)
        producer.join(timeout=5.0)


def infer_regions(
    model,
    frames_dir: Path,
    regions: list[RecoveryRegion],
    device: int,
    artifact_size: res.FrameSize,
    batch_size: int = 32,
    k_best: int = 5,
    peak_threshold: float = 0.05,
    nms_radius: int = 4,
    temporal_ensemble: bool = True,
    subpixel: str = "centroid",
    max_resident_gb: float = 32.0,
    prefetch_batches: int = 2,
    decode_workers: int = 8,
    native_support_audit: dict | None = None,
) -> list[dict]:
    """Compatibility wrapper for callers that explicitly require a materialized list."""
    return [
        row
        for batch in infer_region_batches(
            model,
            frames_dir,
            regions,
            device,
            artifact_size,
            batch_size,
            k_best,
            peak_threshold,
            nms_radius,
            temporal_ensemble,
            subpixel,
            max_resident_gb,
            prefetch_batches,
            decode_workers,
            native_support_audit,
        )
        for row in batch
    ]


def main() -> None:
    # The decode workers are the parallelism; OpenCV's own pool on top of them only burns CPU
    # (the warp and colour conversion are per-pixel, so outputs do not depend on it).
    cv2.setNumThreads(1)
    cuda_blocking_sync.enable()
    baseline.main(
        infer_region_batches=infer_region_batches,
        infer_region_batches_multi=infer_region_batches_multi,
    )


if __name__ == "__main__":
    main()
