"""Native RGB margins for a controlled, optional event-training translation.

Prepare with ``python -m cv.pipeline.event_crop_translation --base-crops BASE
--output OUT --workers 4``. No labels or model outputs are consulted. Every
zero-offset slice must equal its original cached crop before publication.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "event_native_translation_cache_v1"
RADIUS = 4
NATIVE_SIZE = (1920, 1080)
SHAPES = ((192, 108), (384, 216))


def binding(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def margin_origin(left: int, top: int, width: int, height: int) -> tuple[int, int]:
    """An enlarged, entirely native ROI containing every legal radius-4 crop."""
    if not (0 <= left <= 1920 - width and 0 <= top <= 1080 - height):
        raise ValueError("original crop is outside the declared native image")
    return (
        min(max(0, left - RADIUS), 1920 - width - 2 * RADIUS),
        min(max(0, top - RADIUS), 1080 - height - 2 * RADIUS),
    )


def legal_shifts(origins: np.ndarray, radius: int) -> list[tuple[int, int]]:
    if type(radius) is not int or not 0 <= radius <= RADIUS:
        raise ValueError(
            "translation radius must be an integer from zero through four native pixels"
        )
    if np.asarray(origins).shape != (4,):
        raise ValueError("tight and context native origins are required")
    shifts = [
        (dx, dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        if all(
            0 <= origins[2 * channel] + dx <= 1920 - width
            and 0 <= origins[2 * channel + 1] + dy <= 1080 - height
            for channel, (width, height) in enumerate(SHAPES)
        )
    ]
    if (0, 0) not in shifts:
        raise ValueError("original native crop has no valid zero-offset slice")
    return shifts


def shifted_xy(target: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    """Keep the historical center-relative target, then apply the usual flip."""
    return np.asarray(target, dtype=np.float32) - np.asarray(shift, dtype=np.float32) / np.asarray(
        [96, 54], dtype=np.float32
    )


class NativeTranslationStore:
    def __init__(self, directory: Path, base_store) -> None:
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if (
            self.manifest.get("schema") != SCHEMA
            or self.manifest.get("radius_native_px") != RADIUS
            or self.manifest.get("native_image_size") != list(NATIVE_SIZE)
            or self.manifest.get("original_manifest", {}).get("sha256")
            != binding(base_store.directory / "manifest.json")["sha256"]
        ):
            raise ValueError("translation cache does not bind this original native crop inventory")
        if len(self.manifest["shards"]) != len(base_store.shard_paths):
            raise ValueError("translation cache must retain every original shard")
        for prepared, original, source in zip(
            self.manifest["shards"],
            base_store.manifest["shards"],
            base_store.shard_paths,
            strict=True,
        ):
            if (
                prepared["broadcast"] != original["broadcast"]
                or prepared["rows"] != original["rows"]
                or prepared["zero_offset_equal_rows"] != original["rows"]
                or prepared["original_shard"]["sha256"] != binding(source)["sha256"]
            ):
                raise ValueError("translation cache original shard identity or parity changed")
            for name, record in prepared["outputs"].items():
                if (
                    binding(self.directory / prepared["broadcast"] / name)["sha256"]
                    != record["sha256"]
                ):
                    raise ValueError("translation cache output changed")
        self._maps = {}

    def __getstate__(self):
        return {**self.__dict__, "_maps": {}}

    def member(self, shard: int, name: str) -> np.ndarray:
        key = (shard, name)
        if key not in self._maps:
            path = self.directory / self.manifest["shards"][shard]["broadcast"] / name
            self._maps[key] = np.load(path, mmap_mode="r", allow_pickle=False)
        return self._maps[key]

    def sample(self, shard: int, row: int, occurrence: int, *, radius: int, seed: int):
        origins = self.member(shard, "origins.npy")[row]
        margins = self.member(shard, "margin_origins.npy")[row]
        choices = legal_shifts(origins, radius)
        # This local generator never consumes sampler, flip, model or worker RNG.
        generator = np.random.default_rng(np.random.SeedSequence([seed, occurrence, shard, row]))
        dx, dy = choices[int(generator.integers(len(choices)))]
        x, y, xc, yc = origins - margins + np.asarray([dx, dy, dx, dy])
        tight = self.member(shard, "tight.npy")[row, :, y : y + 108, x : x + 192]
        context = self.member(shard, "context.npy")[row, yc : yc + 216, xc : xc + 384]
        if tight.shape != (16, 108, 192, 3) or context.shape != (216, 384, 3):
            raise ValueError("translation cache slice has an invalid native shape")
        return tight, context, (dx, dy)


def _build_shard(arguments: tuple[str, str, int]) -> dict:
    import cv2

    cv2.setNumThreads(1)

    from cv.pipeline.event_crops import (
        FRAMES_DIR,
        _cadence,
        _frame_path,
        crop_bounds,
        sequence_frames,
        unique_exposures,
    )
    from cv.pipeline.event_video_model import CropStore

    base, output, index = arguments
    store = CropStore.open(Path(base))
    metadata = store.manifest["shards"][index]
    broadcast = metadata["broadcast"]
    root = Path(store.manifest["root"]) / broadcast
    destination = Path(output) / broadcast
    destination.mkdir()
    n = metadata["rows"]
    arrays = {
        name: np.lib.format.open_memmap(destination / name, mode="w+", dtype=dtype, shape=shape)
        for name, dtype, shape in (
            ("tight.npy", np.uint8, (n, 16, 116, 200, 3)),
            ("context.npy", np.uint8, (n, 224, 392, 3)),
            ("origins.npy", np.int32, (n, 4)),
            ("margin_origins.npy", np.int32, (n, 4)),
            ("exposures.npy", np.int32, (n, 16)),
        )
    }
    source_before = binding(store.shard_paths[index])
    clips = store.member("clips", index)
    frames = store.member("frames", index)
    centers = store.member("centres", index)
    original_tight = store.member("tight", index)
    original_context = store.member("context", index)
    cadence = _cadence(root)
    pictures = {}
    counts = {}
    for clip in sorted(set(clips.tolist())):
        suffix = str(clip).split("__")[-1]
        files = list((root / FRAMES_DIR / suffix).glob("f_*.jpg"))
        maximum = max((int(p.stem.removeprefix("f_")) for p in files), default=0)
        native_frames = unique_exposures(cadence.get(suffix, []), maximum)
        cache = OrderedDict()

        def image(frame):
            if frame not in cache:
                path = _frame_path(root, suffix, frame)
                raw = path.read_bytes()
                value = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if value is None or value.shape != (1080, 1920, 3):
                    raise ValueError(f"missing or nonnative source picture: {path}")
                digest = hashlib.sha256(raw).hexdigest()
                if str(path) in pictures and pictures[str(path)] != digest:
                    raise ValueError("native source picture changed during margin preparation")
                pictures[str(path)] = digest
                cache[frame] = value
                while len(cache) > 32:
                    cache.popitem(last=False)
            cache.move_to_end(frame)
            return cache[frame]

        for row in np.flatnonzero(clips == clip):
            frame = int(frames[row])
            sequence = sequence_frames(native_frames, frame)
            origins = np.asarray([v for w, h in SHAPES for v in crop_bounds(*centers[row], w, h)])
            margins = np.asarray(
                [
                    v
                    for channel, (width, height) in enumerate(SHAPES)
                    for v in margin_origin(*origins[2 * channel : 2 * channel + 2], width, height)
                ]
            )
            x, y, xc, yc = margins
            for step, member in enumerate(sequence):
                arrays["tight.npy"][row, step] = image(member)[y : y + 116, x : x + 200]
            arrays["context.npy"][row] = image(frame)[yc : yc + 224, xc : xc + 392]
            arrays["origins.npy"][row] = origins
            arrays["margin_origins.npy"][row] = margins
            arrays["exposures.npy"][row] = sequence
            tx, ty, cx, cy = origins - margins
            if not np.array_equal(
                arrays["tight.npy"][row, :, ty : ty + 108, tx : tx + 192], original_tight[row]
            ) or not np.array_equal(
                arrays["context.npy"][row, cy : cy + 216, cx : cx + 384], original_context[row]
            ):
                raise ValueError(f"original crop parity failed: {broadcast} {clip} {frame}")
            choices = len(legal_shifts(origins, RADIUS))
            counts[str(choices)] = counts.get(str(choices), 0) + 1
    for array in arrays.values():
        array.flush()
    if binding(store.shard_paths[index])["sha256"] != source_before["sha256"]:
        raise ValueError("original crop shard changed during margin preparation")
    source_document = {
        "pictures_sha256": pictures,
        "clock": {"original_shard_fps": metadata["fps"], "frames_unchanged": True},
        "timing_artifacts": [
            binding(root / name)
            for name in ("frame_cadence_v1.json", "audit_reel_point_map.csv")
            if (root / name).is_file()
        ],
    }
    (destination / "native_sources.json").write_text(json.dumps(source_document, indent=2) + "\n")
    result = {
        "broadcast": broadcast,
        "rows": n,
        "zero_offset_equal_rows": n,
        "original_shard": source_before,
        "legal_shift_count_rows": counts,
        "outputs": {name: binding(destination / name) for name in (*arrays, "native_sources.json")},
    }
    (destination / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def prepare(base: Path, output: Path, workers: int = 4) -> dict:
    original = binding(base / "manifest.json")
    metadata = json.loads((base / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    jobs = [(str(base), str(output), i) for i in range(len(metadata["shards"]))]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        shards = list(pool.map(_build_shard, jobs))
    if binding(base / "manifest.json")["sha256"] != original["sha256"]:
        raise ValueError("original crop manifest changed during margin preparation")
    result = {
        "schema": SCHEMA,
        "radius_native_px": RADIUS,
        "native_image_size": list(NATIVE_SIZE),
        "original_manifest": original,
        "shards": shards,
        "observations_or_labels_changed": False,
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-crops", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prepare(args.base_crops, args.output, args.workers)


if __name__ == "__main__":
    main()
