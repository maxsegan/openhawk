"""TensorRT fp16 engines for the ball trackers (``TENNIS_BALL_ENGINE=trt_fp16``; default torch).

OUTPUT-CHANGING, near-exact: WASB and TrackNetV2 run as TensorRT fp16 engines in the stage-1
shared tracker (``ball_neural_shared``) and the crop-refine passes (``ball_local_refine_batched``).
Measured in cv/experiments/upstream_dive (FINAL_REPORT_upstream_dive.md §2.3, §2.4): about 2x the
torch fp16 forward with no confident heatmap peak moved against torch fp32. Unset or ``torch``
keeps the torch models (fp32, or fp16 with ``TENNIS_BALL_FP16`` / ``TENNIS_BALL_REFINE_FP16``).

TensorRT never enters the main venv. It lives in one folder per box,
``$TENNIS_DATA_ROOT/tools/<NAME>`` (the cu12 bindings and libraries for the venv's Python),
put on ``sys.path`` only when the switch is on. One command builds it on a box:

    python -m cv.pipeline.ball_trt setup        # install into tools/<NAME> (~6 GB)
    python -m cv.pipeline.ball_trt verify       # import it and print its identity
    python -m cv.pipeline.ball_trt build --device 1   # pre-build this card's engines

``product_boxes setup`` runs all three on every box (engines for each upstream GPU).

Engines are built on first use and cached in ``$TENNIS_DATA_ROOT/tools/ball_trt_engines``
under a key of model, weights sha256, TensorRT and torch versions, GPU compute capability and
name, input size and batch range (one build per box and architecture, ~1-3 min; concurrent
stages wait on a file lock). TensorRT 11 networks are strongly typed, so the engine is built
from an ONNX export of the half-precision model: fp16 in, fp16 logits out, like
``model.half()``. The tactic choice is timed at build time, so two builds (two boxes) may pick
different kernels; their outputs differ at the fp16 rounding level.

When the switch is on but TensorRT cannot load or build, the stage falls back to the torch
fp16 model, says so on stderr and records it (:func:`identity`) in its manifest.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import warnings
from functools import cache
from pathlib import Path

VERSION = "11.3.0.99"
NAME = f"tensorrt-cu12-{VERSION}-cp{sys.version_info.major}{sys.version_info.minor}"
#: Bump when the export or build recipe changes (part of every engine key).
RECIPE = 1
MIN_BATCH, OPT_BATCH, MAX_BATCH = 1, 32, 64
INPUT_CHANNELS = 9
ENGINES = ("trt_fp16",)
#: ``on``: an engine is keyed per architecture (compute capability, versions, weights, recipe),
#: not per card name, so one engine file serves every card of that architecture, and
#: ``product_boxes setup`` ships the dispatcher's engines to each box instead of letting it
#: build its own. A card's own build picks its own tactics: the 5090's crop-refine engines on
#: the local RTX PRO 6000 reproduced the 5090's rows byte for byte, the local build moved 0.4% of
#: them by over 0.5 px (cv/experiments/cross_machine/REPORT.md). Off: per-card keys (old files).
SHARED_ENGINES_ENV = "TENNIS_TRT_SHARED_ENGINES"

_lock = threading.Lock()


def mode() -> str:
    """``torch`` (default) or ``trt_fp16``, from ``TENNIS_BALL_ENGINE``."""
    value = os.environ.get("TENNIS_BALL_ENGINE", "torch") or "torch"
    if value not in ("torch", *ENGINES):
        raise SystemExit(f"TENNIS_BALL_ENGINE={value!r}: expected torch or trt_fp16")
    return value


def data_root() -> Path:
    return Path(os.environ.get("TENNIS_DATA_ROOT", "data"))


def local_dir() -> Path:
    """The TensorRT folder for this Python (override: ``TENNIS_TRT_DIR``)."""
    return Path(os.environ.get("TENNIS_TRT_DIR") or data_root() / "tools" / NAME)


def shared_engines() -> bool:
    return os.environ.get(SHARED_ENGINES_ENV, "off") == "on"


def card_key(device) -> dict:
    """The card name for a per-card engine key; nothing when engines are shared."""
    import torch

    return {} if shared_engines() else {"gpu": torch.cuda.get_device_name(device)}


def engine_file_identity(path: Path, device) -> dict:
    """What an artifact records about the engine file that ran and the card it ran on."""
    import torch

    return {"engine_sha256": _sha256(str(path)), "ran_on": torch.cuda.get_device_name(device)}


def engine_dir() -> Path:
    return Path(
        os.environ.get("TENNIS_TRT_ENGINE_DIR") or data_root() / "tools" / "ball_trt_engines"
    )


def setup(directory: Path | None = None) -> Path:
    """Install the TensorRT bindings and libraries (no other packages) into ``directory``."""
    directory = Path(directory or local_dir())
    staging = directory.with_name(directory.name + ".partial")
    subprocess.run(["rm", "-rf", str(staging)], check=True)
    uv = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
    command = [
        uv, "pip", "install", "--python", sys.executable, "--target", str(staging),
        "--no-deps", "--extra-index-url", "https://pypi.nvidia.com",
        f"tensorrt-cu12-bindings=={VERSION}", f"tensorrt-cu12-libs=={VERSION}",
    ]  # fmt: skip
    subprocess.run(command, check=True)
    subprocess.run(["rm", "-rf", str(directory)], check=True)
    staging.rename(directory)
    return directory


@cache
def tensorrt():
    """The ``tensorrt`` module from :func:`local_dir` (raises ImportError when absent)."""
    directory = local_dir()
    if not (directory / "tensorrt_bindings").is_dir():
        raise ImportError(f"no TensorRT at {directory} (python -m cv.pipeline.ball_trt setup)")
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    module = importlib.import_module("tensorrt_bindings")
    if module.__version__ != VERSION:
        raise ImportError(f"TensorRT {module.__version__} at {directory}, expected {VERSION}")
    return module


@cache
def _logger():
    """One logger per process (TensorRT keeps the first one it is given)."""
    trt = tensorrt()
    return trt.Logger(trt.Logger.WARNING)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def engine_key(model_name: str, weights: str, device: int, input_hw: tuple[int, int]) -> dict:
    import torch

    major, minor = torch.cuda.get_device_capability(device)
    return {
        "model": model_name,
        "weights_sha256": _sha256(weights),
        "tensorrt": VERSION,
        "torch": torch.__version__,
        "compute_capability": f"sm{major}{minor}",
        **card_key(device),
        "input": [INPUT_CHANNELS, *input_hw],
        "batch": [MIN_BATCH, OPT_BATCH, MAX_BATCH],
        "precision": "fp16",
        "recipe": RECIPE,
    }


def _engine_path(key: dict) -> Path:
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    return engine_dir() / f"{key['model']}-{key['compute_capability']}-{digest}.engine"


def _export_onnx(model, path: Path, device: int, input_hw: tuple[int, int]) -> None:
    """Export ``model`` in half precision; it is restored to its exact float32 state after
    (the tracker models cannot be deep-copied)."""
    import torch

    class Logits(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x)[0]

    saved = {name: value.detach().clone() for name, value in model.state_dict().items()}
    training = model.training
    x = torch.zeros((OPT_BATCH, INPUT_CHANNELS, *input_hw), dtype=torch.float16, device=device)
    try:
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # the legacy exporter
            torch.onnx.export(
                # The wrapper must be in eval mode too: the exporter restores the top module's
                # mode recursively, which would leave the tracker in training mode.
                Logits(model.half().eval()).eval(),
                (x,),
                str(path),
                input_names=["x"],
                output_names=["logits"],
                dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=17,
                dynamo=False,
            )
    finally:
        model.float().load_state_dict(saved)
        model.train(training)


def _build(model, key: dict, path: Path, device: int, input_hw: tuple[int, int]) -> None:
    import torch

    trt = tensorrt()
    onnx_path = path.with_suffix(f".{os.getpid()}.onnx")
    try:
        with torch.cuda.device(device):
            _export_onnx(model, onnx_path, device, input_hw)
            logger = _logger()
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
            shape = (INPUT_CHANNELS, *input_hw)
            profile.set_shape("x", (MIN_BATCH, *shape), (OPT_BATCH, *shape), (MAX_BATCH, *shape))
            config.add_optimization_profile(profile)
            serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT build returned no engine")
        partial = path.with_suffix(f".{os.getpid()}.partial")
        partial.write_bytes(bytes(serialized))
        path.with_suffix(".json").write_text(json.dumps(key, indent=1, sort_keys=True) + "\n")
        partial.rename(path)
    finally:
        onnx_path.unlink(missing_ok=True)


def engine_file(model, model_name: str, weights: str, device: int, input_hw) -> tuple[Path, dict]:
    """The cached engine for this model on this card, built (under a file lock) when missing."""
    key = engine_key(model_name, weights, device, tuple(input_hw))
    path = _engine_path(key)
    if path.is_file():
        return path, key
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.is_file():
            started = time.monotonic()
            _build(model, key, path, device, tuple(input_hw))
            print(
                f"ball_trt: built {path.name} in {time.monotonic() - started:.0f} s",
                file=sys.stderr,
                flush=True,
            )
    return path, key


class TrtModel:
    """A tracker as a TensorRT fp16 engine, called like the torch model: ``model(x)[0]`` is
    the (b, 3, H, W) logits (float16) of a (b, 9, H, W) batch of any float dtype or layout.
    Batches above the engine's maximum run in slices. Thread-safe per instance (one execution
    context, one call at a time); give each worker thread its own instance for overlap."""

    def __init__(self, path: Path, key: dict, device: int):
        import torch

        trt = tensorrt()
        self.torch = torch
        self.key = key
        self.path = path
        self.device = torch.device("cuda", device)
        with torch.cuda.device(device):
            runtime = trt.Runtime(_logger())
            self.engine = runtime.deserialize_cuda_engine(path.read_bytes())
            if self.engine is None:
                raise RuntimeError(f"cannot load engine {path}")
            self.context = self.engine.create_execution_context()
            self.file = engine_file_identity(path, device)
            # TensorRT synchronises extra on the legacy default stream: run on our own stream,
            # ordered after the caller's work and before anything the caller does next.
            self.stream = torch.cuda.Stream(self.device)
        self._call = threading.Lock()

    def __call__(self, x):
        torch = self.torch
        x = x.to(self.device, torch.float16).contiguous()
        out = torch.empty((len(x), 3, *x.shape[2:]), dtype=torch.float16, device=self.device)
        caller = torch.cuda.current_stream(self.device)
        with self._call:
            self.stream.wait_stream(caller)
            for start in range(0, len(x), MAX_BATCH):
                part = x[start : start + MAX_BATCH]
                self.context.set_input_shape("x", tuple(part.shape))
                self.context.set_tensor_address("x", part.data_ptr())
                self.context.set_tensor_address("logits", out[start : start + MAX_BATCH].data_ptr())
                if not self.context.execute_async_v3(self.stream.cuda_stream):
                    raise RuntimeError("TensorRT execution failed")
            caller.wait_stream(self.stream)
        return (out,)


def identity(model) -> dict:
    """What ran: ``{"engine": "torch"}`` or the TensorRT engine key and file."""
    if isinstance(model, TrtModel):
        return {"engine": "trt_fp16", "engine_file": model.path.name, **model.key, **model.file}
    return {"engine": getattr(model, "_ball_engine", "torch")}


def wrap(model, model_name: str, weights: str, device: int, input_hw=(288, 512)):
    """``model`` as a :class:`TrtModel` when ``TENNIS_BALL_ENGINE=trt_fp16`` (else unchanged).

    On any TensorRT failure the torch model is returned in half precision and channels-last
    (the ``--fp16`` path), marked ``_ball_engine = "torch_fp16 (fallback: ...)"``."""
    if mode() == "torch":
        return model
    try:
        with _lock:
            path, key = engine_file(model, model_name, weights, device, input_hw)
        return TrtModel(path, key, device)
    except Exception as error:  # noqa: BLE001 - any TensorRT failure falls back to torch
        import torch

        from cv.pipeline import switch_fallbacks

        reason = f"{type(error).__name__}: {error}"[:500]
        print(f"ball_trt: TensorRT unavailable, torch fp16 fallback ({reason})", file=sys.stderr)
        kind = "tensorrt_missing" if isinstance(error, ImportError) else "engine_unavailable"
        switch_fallbacks.record("TENNIS_BALL_ENGINE", kind, reason, where=model_name)
        model = model.half().to(memory_format=torch.channels_last)
        model._ball_engine = f"torch_fp16 (fallback: {reason})"
        return model


def is_half(model) -> bool:
    """True when ``model`` expects fp16 inputs (a TensorRT engine or the torch fallback)."""
    return isinstance(model, TrtModel) or str(getattr(model, "_ball_engine", "")).startswith(
        "torch_fp16"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("action", choices=("setup", "verify", "build"))
    parser.add_argument("--dir", type=Path, default=None, help="TensorRT folder (default: tools)")
    parser.add_argument("--device", type=int, default=0, help="build: CUDA device")
    args = parser.parse_args()
    if args.dir is not None:
        os.environ["TENNIS_TRT_DIR"] = str(args.dir)
    if args.action == "setup":
        print(setup(args.dir))
        return 0
    trt = tensorrt()
    if args.action == "verify":
        print(json.dumps({"dir": str(local_dir()), "tensorrt": trt.__version__}))
        return 0
    from cv.pipeline import ball_neural
    from cv.pipeline.paths import tracker_root

    from cv.pipeline.tracking_composition import CROP_FINETUNE_TEMPLATE

    root = os.fspath(tracker_root())
    os.environ["TENNIS_BALL_ENGINE"] = "trt_fp16"
    for name in ("wasb", "tracknetv2"):
        # Stage-1 (sliding and far) and the crop-refine fine-tunes, when present.
        for weights in (
            os.path.join(root, "pretrained_weights", f"{name}_tennis_best.pth.tar"),
            os.fspath(data_root() / CROP_FINETUNE_TEMPLATE.format(model=name)),
        ):
            if not os.path.isfile(weights):
                print(json.dumps({"missing": weights}))
                continue
            model = ball_neural.build_model(root, name, weights, args.device)
            path, key = engine_file(model, name, weights, args.device, (288, 512))
            print(json.dumps({"engine": str(path), **key}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
