"""TensorRT fp16 engine for player detection (``TENNIS_PLAYER_ENGINE=trt_fp16``; default torch).

OUTPUT-CHANGING, near-exact. On a free RTX PRO 6000 at its 300 W cap, player detection
(YOLOv8m at imgsz 1920, letterboxed to 1088x1920) runs at ~85% SM: the stage is compute-bound,
not feed-bound, so the remaining lever is less arithmetic per frame. The engine is built from
ultralytics' own half-precision ONNX export (raw ``(b, 84, anchors)`` predictions, the tensor
``predictor.inference`` returns), so ultralytics' preprocessing, NMS and box scaling are
unchanged; only the network runs in fp16. Measured in cv/experiments/feed_gpus (REPORT.md).

TensorRT itself, its folder and the engine cache are :mod:`cv.pipeline.ball_trt`'s. Engines are
keyed by weights sha256, TensorRT/torch/ultralytics versions, GPU, input size and batch range,
built on first use under a file lock. Any TensorRT failure falls back to the torch model (fp32,
the default path), recorded through :mod:`cv.pipeline.switch_fallbacks`.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

from cv.pipeline import ball_trt

ENV = "TENNIS_PLAYER_ENGINE"
#: Bump when the export or build recipe changes (part of every engine key).
RECIPE = 1
MIN_BATCH, OPT_BATCH, MAX_BATCH = 1, 64, 64

_lock = threading.Lock()


def mode() -> str:
    value = os.environ.get(ENV, "torch") or "torch"
    if value not in ("torch", "trt_fp16"):
        raise SystemExit(f"{ENV}={value!r}: expected torch or trt_fp16")
    return value


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def engine_key(weights: str, device: int, input_hw: tuple[int, int]) -> dict:
    import torch
    import ultralytics

    major, minor = torch.cuda.get_device_capability(device)
    return {
        "model": "yolo_players",
        "weights_sha256": _sha256(weights),
        "tensorrt": ball_trt.VERSION,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "compute_capability": f"sm{major}{minor}",
        **ball_trt.card_key(device),
        "input": [3, *input_hw],
        "batch": [MIN_BATCH, OPT_BATCH, MAX_BATCH],
        "precision": "fp16",
        "recipe": RECIPE,
    }


def _build(weights: str, key: dict, path: Path, device: int, input_hw: tuple[int, int]) -> None:
    import torch
    from ultralytics import YOLO

    trt = ball_trt.tensorrt()
    with tempfile.TemporaryDirectory(dir=path.parent) as scratch:
        # The exporter writes next to its weights: export a private copy.
        local = Path(scratch) / Path(weights).name
        shutil.copy2(weights, local)
        with torch.cuda.device(device):
            onnx_path = YOLO(str(local)).export(
                format="onnx",
                imgsz=list(input_hw),
                half=True,
                dynamic=True,
                device=device,
                simplify=False,
                opset=17,
                verbose=False,
            )
            logger = ball_trt._logger()
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
            )
            parser = trt.OnnxParser(network, logger)
            if not parser.parse_from_file(str(onnx_path)):
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError(f"ONNX parse failed: {errors}")
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
            profile = builder.create_optimization_profile()
            name = network.get_input(0).name
            shape = (3, *input_hw)
            profile.set_shape(name, (MIN_BATCH, *shape), (OPT_BATCH, *shape), (MAX_BATCH, *shape))
            config.add_optimization_profile(profile)
            serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT build returned no engine")
        partial = path.with_suffix(f".{os.getpid()}.partial")
        partial.write_bytes(bytes(serialized))
        path.with_suffix(".json").write_text(json.dumps(key, indent=1, sort_keys=True) + "\n")
        partial.rename(path)


def engine_file(weights: str, device: int, input_hw: tuple[int, int]) -> tuple[Path, dict]:
    key = engine_key(weights, device, input_hw)
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    path = ball_trt.engine_dir() / f"{key['model']}-{key['compute_capability']}-{digest}.engine"
    if path.is_file():
        return path, key
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.is_file():
            started = time.monotonic()
            _build(weights, key, path, device, input_hw)
            print(
                f"player_trt: built {path.name} in {time.monotonic() - started:.0f} s",
                file=sys.stderr,
                flush=True,
            )
    return path, key


class TrtDetector:
    """``predictor.inference`` for a letterboxed ``(b, 3, H, W)`` batch: the raw fp16
    ``(b, 84, anchors)`` predictions. One execution context; one call at a time."""

    def __init__(self, path: Path, key: dict, device):
        import torch

        trt = ball_trt.tensorrt()
        self.torch = torch
        self.key = key
        self.path = path
        self.device = torch.device(device)
        with torch.cuda.device(self.device):
            runtime = trt.Runtime(ball_trt._logger())
            self.engine = runtime.deserialize_cuda_engine(path.read_bytes())
            if self.engine is None:
                raise RuntimeError(f"cannot load engine {path}")
            self.context = self.engine.create_execution_context()
            self.stream = torch.cuda.Stream(self.device)
        self.file = ball_trt.engine_file_identity(path, self.device)
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self._call = threading.Lock()

    def __call__(self, x):
        torch = self.torch
        x = x.to(self.device, torch.float16).contiguous()
        caller = torch.cuda.current_stream(self.device)
        outputs = []
        with self._call:
            self.stream.wait_stream(caller)
            for start in range(0, len(x), MAX_BATCH):
                part = x[start : start + MAX_BATCH]
                self.context.set_input_shape(self.names[0], tuple(part.shape))
                out = torch.empty(
                    tuple(self.context.get_tensor_shape(self.names[1])),
                    dtype=torch.float16,
                    device=self.device,
                )
                self.context.set_tensor_address(self.names[0], part.data_ptr())
                self.context.set_tensor_address(self.names[1], out.data_ptr())
                if not self.context.execute_async_v3(self.stream.cuda_stream):
                    raise RuntimeError("TensorRT execution failed")
                outputs.append(out)
            caller.wait_stream(self.stream)
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs)

    def identity(self) -> dict:
        return {"engine": "trt_fp16", "engine_file": self.path.name, **self.key, **self.file}


def load(weights: str, device, input_hw: tuple[int, int]) -> TrtDetector | None:
    """The engine for ``weights`` on ``device`` (a torch device), or None (torch path) when the
    switch is off or TensorRT fails (recorded as a switch fallback)."""
    if mode() == "torch":
        return None
    import torch

    index = torch.device(device).index or 0
    try:
        with _lock:
            path, key = engine_file(str(weights), index, tuple(input_hw))
        return TrtDetector(path, key, torch.device("cuda", index))
    except Exception as error:  # noqa: BLE001 - any TensorRT failure keeps the torch model
        from cv.pipeline import switch_fallbacks

        reason = f"{type(error).__name__}: {error}"[:500]
        print(f"player_trt: TensorRT unavailable, torch fp32 kept ({reason})", file=sys.stderr)
        kind = "tensorrt_missing" if isinstance(error, ImportError) else "engine_unavailable"
        switch_fallbacks.record(ENV, kind, reason, where="player_detection")
        return None
