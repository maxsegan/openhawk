"""GPU guard for the product runner's 3D pose step: one at a time per box, free memory, OOM retry.

The ``pose`` step (``cv.pipeline.player_pose_3d``) runs on GPU 0 beside the 55 GB scoreboard
VLM and peaks at ~20 GB. Two pose steps at once (two queue drivers share the box, each with
its own light slots), or one beside other GPU-0 work, ran out of CUDA memory and the match
published without pose (fix-list 27, 2026-10-09). :func:`guarded_run` therefore:

1. takes a box-wide ``flock`` (``<dir>/pose.lock``; default ``DATA/processed/box_slots/
   local-<hostname>``, beside the queue's upstream slots, so every driver on the machine
   shares it and the kernel releases it if a runner dies): at most one pose step at a time;
2. before each attempt, waits until the card has ``TENNIS_POSE_FREE_GB`` free (nvidia-smi;
   at most ``TENNIS_POSE_FREE_WAIT_S``, then the attempt runs anyway and the timeout is
   recorded; a card nvidia-smi does not report is taken as is);
3. retries an attempt that failed with CUDA out of memory, after ``TENNIS_POSE_OOM_BACKOFF_S``
   times the attempt number, up to ``TENNIS_POSE_OOM_ATTEMPTS`` attempts. Every attempt is
   recorded; when every attempt ran out of memory, :class:`PoseOutOfMemory` is raised so the
   step fails (and the match is not compacted or published without pose).

Switches (recorded in the step detail): ``TENNIS_POSE_GPU_LOCK_DIR`` (``off``: no lock),
``TENNIS_POSE_FREE_GB`` (22), ``TENNIS_POSE_FREE_WAIT_S`` (1800), ``TENNIS_POSE_OOM_ATTEMPTS``
(3), ``TENNIS_POSE_OOM_BACKOFF_S`` (120). Throughput only: pose outputs are unchanged.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import socket
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from cv.pipeline.gpu_lease import probe_gpus

ENV_LOCK_DIR = "TENNIS_POSE_GPU_LOCK_DIR"
ENV_FREE_GB = "TENNIS_POSE_FREE_GB"
ENV_FREE_WAIT_S = "TENNIS_POSE_FREE_WAIT_S"
ENV_ATTEMPTS = "TENNIS_POSE_OOM_ATTEMPTS"
ENV_BACKOFF_S = "TENNIS_POSE_OOM_BACKOFF_S"
#: Pose peaks ~20 GB (production 2026-10-09); a little headroom over it.
DEFAULT_FREE_GB = 22.0
DEFAULT_FREE_WAIT_S = 1800.0
DEFAULT_ATTEMPTS = 3
DEFAULT_BACKOFF_S = 120.0
POLL_SECONDS = 15.0
LOCK_NAME = "pose.lock"
#: Log text of a CUDA allocation failure (torch, the CUDA runtime/driver, nvJPEG).
OOM_MARKERS = (
    "CUDA out of memory",
    "OutOfMemoryError",
    "CUDA error: out of memory",
    "CUDA_ERROR_OUT_OF_MEMORY",
    "cudaErrorMemoryAllocation",
    "NVJPEG_STATUS_ALLOCATOR_FAILURE",
)


class PoseOutOfMemory(RuntimeError):
    """Every pose attempt ran out of GPU memory."""


def default_lock_dir() -> Path:
    data = Path(os.environ.get("TENNIS_DATA_ROOT", "data"))
    # Same key as product_queue.BoxSlots for this machine.
    return data / "processed" / "box_slots" / f"local-{socket.gethostname()}"


def settings() -> dict:
    """The guard's switches now (env or default)."""
    lock = os.environ.get(ENV_LOCK_DIR) or str(default_lock_dir())
    return {
        "lock_dir": None if lock == "off" else lock,
        "free_gb": float(os.environ.get(ENV_FREE_GB) or DEFAULT_FREE_GB),
        "free_wait_s": float(os.environ.get(ENV_FREE_WAIT_S) or DEFAULT_FREE_WAIT_S),
        "attempts": max(1, int(os.environ.get(ENV_ATTEMPTS) or DEFAULT_ATTEMPTS)),
        "backoff_s": float(os.environ.get(ENV_BACKOFF_S) or DEFAULT_BACKOFF_S),
    }


def is_oom(text: str) -> bool:
    return any(marker in text for marker in OOM_MARKERS)


@contextlib.contextmanager
def box_lock(lock_dir: str | Path | None) -> Iterator[dict]:
    """Hold ``<lock_dir>/pose.lock`` exclusively (blocking); yields the wait record."""
    if lock_dir is None:
        yield {"lock": None}
        return
    started = time.time()
    path = Path(lock_dir) / LOCK_NAME
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a")
    except OSError:
        # An unwritable data root: every runner on this machine falls back to the same file.
        path = Path(tempfile.gettempdir()) / "tennis-box-locks" / LOCK_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield {"lock": str(path), "lock_wait_s": round(time.time() - started, 1)}
    finally:
        handle.close()  # closing the descriptor drops its flock


def wait_for_free(
    gpu: int | None,
    need_gb: float,
    max_wait_s: float,
    *,
    probe: Callable[[], dict[int, dict]] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
    poll_s: float = POLL_SECONDS,
) -> dict:
    """Wait until physical ``gpu`` has ``need_gb`` free; a record of the wait."""
    if gpu is None or need_gb <= 0:
        return {"free_check": "off"}
    probe = probe or probe_gpus
    started = clock()
    polls = 0
    while True:
        status = (probe() or {}).get(int(gpu))
        if status is None:
            return {"free_check": "no_probe", "free_wait_s": round(clock() - started, 1)}
        free_gb = (status["mem_total_mib"] - status["mem_used_mib"]) / 1024
        waited = clock() - started
        if free_gb >= need_gb:
            return {
                "free_check": "ok",
                "free_gb": round(free_gb, 1),
                "free_wait_s": round(waited, 1),
                "free_polls": polls,
            }
        if waited >= max_wait_s:
            return {
                "free_check": "timeout",
                "free_gb": round(free_gb, 1),
                "free_wait_s": round(waited, 1),
                "free_polls": polls,
            }
        polls += 1
        sleep(poll_s)


def guarded_run(
    attempt: Callable[[], tuple[int, str]],
    *,
    gpu: int | None,
    config: dict | None = None,
    record: Path | None = None,
    probe: Callable[[], dict[int, dict]] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[int, list[dict], dict]:
    """Run ``attempt`` (returns ``(returncode, log text of this attempt)``) under the guard.

    ``gpu``: the physical card to check free memory on (None: no check, e.g. when a lease
    picks the card). Returns the last return code, every attempt's record and the settings.
    Each attempt record is also appended to ``record`` (JSON lines) as it ends. Raises
    :class:`PoseOutOfMemory` when every attempt ran out of memory.
    """
    config = config or settings()
    attempts: list[dict] = []
    code = 0
    with box_lock(config["lock_dir"]) as held:
        for number in range(1, config["attempts"] + 1):
            entry: dict = {"attempt": number, **(held if number == 1 else {})}
            entry |= wait_for_free(
                gpu, config["free_gb"], config["free_wait_s"], probe=probe, sleep=sleep
            )
            started = time.time()
            code, text = attempt()
            oom = code != 0 and is_oom(text)
            entry |= {
                "started": round(started, 1),
                "seconds": round(time.time() - started, 1),
                "returncode": code,
                "oom": oom,
            }
            last = number == config["attempts"]
            backoff = config["backoff_s"] * number if oom and not last else 0.0
            if backoff:
                entry["backoff_s"] = backoff
            attempts.append(entry)
            if record is not None:
                with contextlib.suppress(OSError):
                    record.parent.mkdir(parents=True, exist_ok=True)
                    with record.open("a") as handle:
                        handle.write(json.dumps(entry) + "\n")
            if not oom:
                return code, attempts, config
            if backoff:
                sleep(backoff)
    raise PoseOutOfMemory(
        f"player_pose: CUDA out of memory on all {len(attempts)} attempts "
        f"(gpu {gpu}, need {config['free_gb']} GB free): {json.dumps(attempts)}"
    )
