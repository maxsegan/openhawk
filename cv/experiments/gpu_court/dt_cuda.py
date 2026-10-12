"""One-launch CUDA version of ``torch_distance.distance_transform_tensor`` (same integer chamfer).

The row loop in PyTorch issues ~10 small kernels per image row, so it is launch-bound and costs
about as much CPU as OpenCV's transform. ``dt_kernel.cu`` runs both raster passes of a whole
batch in one launch: one block per image, with exact int64 min-plus scans across each row.
It is built with nvcc into ``TENNIS_COURT_GPU_BUILD`` (default
``$TENNIS_DATA_ROOT/processed/gpu-court/build``) once per source hash, under a file lock.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import subprocess
from pathlib import Path

import torch

from cv.experiments.gpu_court import torch_distance

SOURCE = Path(__file__).with_name("dt_kernel.cu")
NVCC = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
_LIBRARY = None


def _build_dir() -> Path:
    if os.environ.get("TENNIS_COURT_GPU_BUILD"):
        return Path(os.environ["TENNIS_COURT_GPU_BUILD"])
    root = Path(os.environ.get("TENNIS_DATA_ROOT", "data"))
    return root / "processed" / "gpu-court" / "build"


#: The court kernels the queue prebuilds here and pushes to boxes without nvcc
#: (``product_boxes.push_kernel``): (kernel source, prefix, extra nvcc flags).
KERNELS = (
    (SOURCE, "dt_kernel", ()),
    (Path(__file__).with_name("sift_kernel.cu"), "sift_kernel", ("--fmad=false",)),
)


def build(
    source: Path,
    prefix: str,
    extra_flags: tuple[str, ...] = (),
    capability: tuple[int, int] | None = None,
) -> ctypes.CDLL | Path:
    """Compile ``source`` once per source/flags hash (file lock across processes) and load it.

    With ``capability`` (another card's ``(major, minor)``) it only builds and returns the path:
    the same name the card's own process looks for, so a box without nvcc loads it as is.
    """
    major, minor = capability or torch.cuda.get_device_capability()
    flags = ["-O3", "-std=c++17", f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"]
    flags += [*extra_flags, "-shared", "-Xcompiler", "-fPIC"]
    digest = hashlib.sha256(source.read_bytes() + " ".join(flags).encode()).hexdigest()[:16]
    target = _build_dir() / f"{prefix}_{digest}.so"
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not target.is_file():
                partial = target.with_suffix(f".{os.getpid()}.tmp")
                subprocess.run([NVCC, *flags, str(source), "-o", str(partial)], check=True)
                os.replace(partial, target)
    return target if capability else ctypes.CDLL(str(target))


def prebuild(capabilities) -> list[Path]:
    """Every court kernel for each ``(major, minor)`` (no GPU needed, nvcc here)."""
    return [build(*kernel, capability=tuple(c)) for c in capabilities for kernel in KERNELS]


def library():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    lib = build(SOURCE, "dt_kernel")
    p = ctypes.c_void_p
    lib.chamfer_batch.restype = ctypes.c_int
    lib.chamfer_batch.argtypes = [p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.chamfer_batch.argtypes += [ctypes.c_longlong] * 3 + [p, p, ctypes.c_float]
    _LIBRARY = lib
    return lib


@torch.no_grad()
def distance_transform_tensor(src: torch.Tensor) -> torch.Tensor:
    """(B, H, W) CUDA uint8 sources (nonzero = background) -> (B, H, W) float32 distances."""
    src = (src != 0).to(torch.uint8).contiguous()
    B, H, W = src.shape
    scratch = torch.empty((B, H, W), dtype=torch.int32, device=src.device)
    out = torch.empty((B, H, W), dtype=torch.float32, device=src.device)
    torch.cuda.set_device(src.device)
    rc = library().chamfer_batch(
        src.data_ptr(),
        B,
        H,
        W,
        torch_distance.HV,
        torch_distance.DIAG,
        torch_distance.DIST_MAX,
        scratch.data_ptr(),
        out.data_ptr(),
        float(torch_distance.SCALE),
    )
    if rc:
        raise RuntimeError(f"chamfer kernel failed ({rc})")
    return out
