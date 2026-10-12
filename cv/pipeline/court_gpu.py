"""The court GPU package as one match-level choice: reserve, wait, retry, or all on the CPU.

Court geometry's GPU package (``TENNIS_COURT_SIFT=gpu|cuda``, ``TENNIS_COURT_SUPPORT=gpu``,
``TENNIS_COURT_DECODE=nvjpeg``) runs in every court worker process. GPU and CPU court paths give
cameras 1-4 px apart, so a match must never mix them (dry run 2, 2026-10-02: workers that hit
CUDA OOM fell back one call at a time). Here:

- the step's GPU lease reserves :func:`need_gb` for its workers (``canonical_runner``), measured
  per worker, and waits until the card has it;
- an out-of-memory error after that is retried on the GPU (:func:`retry_on_oom`: release this
  process's cache, wait, try again) instead of switching path;
- any other GPU failure, or OOM that outlasts :data:`OOM_WAIT_SECONDS`, raises
  :class:`CourtGpuUnavailable`; ``court_topology_runner`` then reruns the whole match's court
  step with :data:`CPU_SWITCHES` and records that fallback loudly, once for the match.

nvJPEG decode is byte-identical to ``cv2.imread``, so a decoder that cannot open on a card (no
NVJPG engine on GeForce) still decodes with cv2 per process; only its OOM is retried here.
"""

from __future__ import annotations

import os
import random
import re
import sys
import time
from collections.abc import Callable
from typing import TypeVar

SIFT = "TENNIS_COURT_SIFT"
SUPPORT = "TENNIS_COURT_SUPPORT"
DECODE = "TENNIS_COURT_DECODE"
DEVICE = "TENNIS_COURT_SIFT_DEVICE"
#: The whole match's court step without the GPU package.
CPU_SWITCHES = {SIFT: "cpu", SUPPORT: "cpu", DECODE: "cv2"}
#: GPU memory one court worker process needs (GB). Measured 2026-10-03 on the RTX PRO 6000
#: (48 Kontaveit-Swiatek clips, 8 workers, full package with magsac): 4.35-4.57 GiB per
#: process in nvidia-smi (CUDA context plus PyTorch cache), 35.3 GiB for the 8; dry run 2's
#: OOMing workers held 4.0-4.4 GB. Rounded up for clips with more frames per chunk.
PROCESS_GB = 5.0
#: How long one GPU call keeps retrying out-of-memory before the match goes to the CPU.
OOM_WAIT_SECONDS = 1800.0
OOM_FIRST_WAIT_SECONDS = 5.0
OOM_MAX_WAIT_SECONDS = 60.0

T = TypeVar("T")


class CourtGpuUnavailable(RuntimeError):
    """A court GPU switch cannot run in this process: the match's court step goes to the CPU."""

    def __init__(self, switch: str, kind: str, detail: str) -> None:
        super().__init__(switch, kind, detail)
        self.switch, self.kind, self.detail = switch, kind, detail

    def __str__(self) -> str:
        return f"{self.switch} {self.kind}: {self.detail}"


def on(environ=None) -> bool:
    """The GPU package is asked for (SIFT or support on the GPU; decode alone is neutral)."""
    environ = os.environ if environ is None else environ
    return environ.get(SIFT, "cpu") in ("gpu", "cuda") or environ.get(SUPPORT, "cpu") == "gpu"


def need_gb(jobs: int) -> float:
    """Free GPU memory the court step's lease waits for: one :data:`PROCESS_GB` per worker."""
    return round(PROCESS_GB * max(1, int(jobs)), 1)


def card_total_gb(probe: Callable[[], dict[int, dict]] | None = None) -> float | None:
    """Memory (GB) of the card the court workers use (``TENNIS_COURT_SIFT_DEVICE`` among
    ``CUDA_VISIBLE_DEVICES``), from nvidia-smi without a CUDA context; None when unknown."""
    from cv.pipeline import gpu_lease

    device = int(os.environ.get(DEVICE, "0"))
    visible = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v]
    if visible:
        if device >= len(visible) or not visible[device].isdigit():
            return None
        device = int(visible[device])
    status = (probe or gpu_lease.probe_gpus)().get(device)
    return None if status is None else status["mem_total_mib"] / 1024


def gpu_workers(jobs: int, total_gb: float | None, reserve_gb: float = 2.0) -> int:
    """Court workers on the GPU path: ``jobs``, but no more than the card holds at
    :data:`PROCESS_GB` each (the 32 GB RTX 5090 holds 5 of the 8 a match asks for). Clips are
    independent, so the worker count does not change outputs."""
    jobs = max(1, int(jobs))
    if total_gb is None:
        return jobs
    return max(1, min(jobs, int((total_gb - reserve_gb) // PROCESS_GB)))


def is_out_of_memory(error: BaseException) -> bool:
    """CUDA, cuDNN, cuBLAS or nvJPEG ran out of memory (cuDNN reports it as an internal error;
    seen under dry run 2's OOM)."""
    if type(error).__name__ == "OutOfMemoryError":
        return True
    text = str(error)
    if re.search(r"nvJPEG .* failed with status 5\b", text):  # NVJPEG_STATUS_ALLOCATOR_FAILURE
        return True
    return any(
        marker in text
        for marker in (
            "out of memory",
            "CUDNN_STATUS_INTERNAL_ERROR",
            "CUDNN_STATUS_ALLOC_FAILED",
            "CUBLAS_STATUS_ALLOC_FAILED",
        )
    )


def _release_cache() -> None:
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass


def retry_on_oom(
    switch: str,
    call: Callable[[], T],
    *,
    wait_seconds: float = OOM_WAIT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> T:
    """``call()``, retried on the GPU after out-of-memory until ``wait_seconds`` have passed.

    Other errors propagate unchanged (the caller decides what they mean). Between tries this
    process's PyTorch cache is released so co-resident workers can make progress.
    """
    deadline = clock() + wait_seconds
    pause = OOM_FIRST_WAIT_SECONDS
    tries = 0
    while True:
        try:
            return call()
        except Exception as error:  # noqa: BLE001 - classified below, others re-raised
            if not is_out_of_memory(error):
                raise
            tries += 1
            if clock() + pause > deadline:
                raise CourtGpuUnavailable(
                    switch,
                    "out_of_memory",
                    f"{type(error).__name__}: {error}"[:300] + f" (after {tries} tries)",
                ) from error
            print(
                f"WARNING court GPU: {switch} out of memory (try {tries}, pid {os.getpid()}); "
                f"retrying on the GPU in {pause:.0f} s",
                file=sys.stderr,
                flush=True,
            )
            _release_cache()
            sleep(pause * random.uniform(0.75, 1.25))
            pause = min(pause * 2, OOM_MAX_WAIT_SECONDS)
