"""Stage-level GPU leases shared by every match on one machine (throughput only).

A match's upstream holds a GPU only while one of its GPU stages runs (player detection,
tracking, event inference, ...), not through its CPU stages (court calibration, point gates,
audio, frame extraction). Leases are ``flock`` locks on unit files in one directory per
machine, so they are shared across processes, need no server, and are released by the
kernel when a holder dies.

The queue (``cv.pipeline.product_queue --gpu-leasing``) passes the lease specification to
``product_runner --gpu-lease SPEC``, which exports it as ``PRODUCT_GPU_LEASE``; code that
runs a GPU stage wraps it in :func:`environment` or :func:`run`. Without the variable both
are no-ops and the stage uses the process's own ``CUDA_VISIBLE_DEVICES`` (behaviour before
leasing). Commands keep ``--device 0``; the lease maps device 0 to the leased physical GPU
through ``CUDA_VISIBLE_DEVICES``, so stage commands and their resume fingerprints are the
same with and without leasing.

Specification (JSON)::

    {"dir": ".../_gpu_leases", "gpus": {"1": 4, "2": 4, "3": 4},
     "stage_units": {"tracking": 2}, "default_units": 1,
     "hot_c": 83, "min_free_gb_per_unit": 22, "log": ".../gpu_leases.jsonl"}

``gpus`` is each GPU's capacity in units; a stage takes ``stage_units.get(stage,
default_units)`` units (capped at the card's capacity) of ONE GPU. Choice among GPUs with
enough free units: skip a GPU without ``min_free_gb_per_unit * units`` of free memory
(other lanes' jobs count) or, when the spec has ``stage_free_gb`` (per-box measured needs,
``product_boxes.STAGE_FREE_GB``), without ``stage_free_gb.get(stage, default_free_gb)``
capped at the card's size less ``reserve_gb`` (a small card then waits until it is all but
empty), prefer cool over hot (above ``hot_c`` or thermally throttled; a
hot GPU takes a lease only when none of its units is held), then fewest held units, lowest
utilisation, lowest temperature. A spec with ``gate`` (a path) gets no lease while that file
is absent (``--cohorts``: GPU priority by cohort, ``cv.pipeline.queue_cohorts``). A lease held by this process tree is not taken twice
(``PRODUCT_GPU_LEASE_HELD``).

Light cards (``light``, lane C 2026-10-03; off without it): a card listed there, e.g. GPU 0
beside the scoreboard VLM, takes only its listed ``stages``, only when no ordinary card can,
and never while it is hot (above its own ``hot_c`` or thermally throttled, held units or
not), while its free memory less its ``reserve_gb`` is below the stage's need, or while its
``guard`` file (written by the queue's VLM health probe, ``cv.pipeline.queue_feed``) is
missing, older than ``guard_max_age_s`` or not ``ok``::

    "light": {"0": {"stages": ["event_inference", ...], "hot_c": 80, "reserve_gb": 4,
                    "guard": ".../gpu0.guard.json", "guard_max_age_s": 180}}

A process waiting for a lease holds a lock on ``<dir>/waiting/<pid>.<stage>.lock`` (see
:func:`waiters`; the queue's ``--admission feed`` admits no match while one waits). With
``sample_seconds`` in the spec each lease samples its card while held and logs the peak
temperature, the lowest SM clock and whether it thermally throttled.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

ENV_SPEC = "PRODUCT_GPU_LEASE"
ENV_HELD = "PRODUCT_GPU_LEASE_HELD"
ENV_LOG = "PRODUCT_GPU_LEASE_LOG"

#: nvidia-smi ``clocks_throttle_reasons.active`` bits that mean heat, not the power cap
#: every loaded card sits at (0x4): SW thermal (0x20), HW slowdown (0x8), HW thermal (0x40),
#: HW power brake (0x80).
THERMAL_THROTTLE_BITS = 0x08 | 0x20 | 0x40 | 0x80
GPU_QUERY = (
    "index,temperature.gpu,utilization.gpu,memory.used,memory.total,clocks.sm,"
    "clocks_throttle_reasons.active"
)


def probe_gpus(timeout: float = 20.0) -> dict[int, dict]:
    """This machine's GPUs now: temperature, utilisation, memory, SM clock, throttle bits.

    ``{}`` when nvidia-smi is unavailable (callers then fall back to plain rotation).
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    gpus = {}
    for line in out.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 7:
            continue
        try:
            index, temp, util, used, total, sm = (int(float(f)) for f in fields[:6])
            throttle = int(fields[6], 16)
        except ValueError:
            continue
        gpus[index] = {
            "temp_c": temp,
            "util_percent": util,
            "mem_used_mib": used,
            "mem_total_mib": total,
            "sm_mhz": sm,
            "throttle": throttle,
        }
    return gpus


def gpu_hot(status: dict, hot_c: float) -> bool:
    return status["temp_c"] > hot_c or bool(status["throttle"] & THERMAL_THROTTLE_BITS)


def spec_from_env() -> dict | None:
    text = os.environ.get(ENV_SPEC)
    return json.loads(text) if text else None


class Lease:
    """Held unit locks on one GPU; :meth:`release` (or process exit) frees them."""

    def __init__(self, gpu: str, units: int, handles: list) -> None:
        self.gpu, self.units, self._handles = gpu, units, handles

    def release(self) -> None:
        for handle in self._handles:
            with contextlib.suppress(OSError):
                fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
        self._handles = []


def _unit_path(directory: Path, gpu: str, unit: int) -> Path:
    return directory / f"gpu{gpu}.unit{unit}.lock"


def _try_lock(path: Path):
    handle = path.open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def held_units(directory: Path, gpu: str, capacity: int) -> int:
    """Units of ``gpu`` held by some process now (probe-locks each free one briefly)."""
    held = 0
    for unit in range(capacity):
        handle = _try_lock(_unit_path(directory, gpu, unit))
        if handle is None:
            held += 1
        else:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
    return held


def stage_need_mib(spec: dict, stage: str, total_mib: float, need_gb: float | None = None) -> float:
    """Free MiB ``stage`` needs on a card of ``total_mib`` (0 without ``stage_free_gb``).

    ``need_gb``: the caller's own measured need (court geometry: per worker times workers),
    checked on every box whether or not its spec has ``stage_free_gb``.
    """
    needs = spec.get("stage_free_gb")
    if need_gb is None and not needs:
        return 0.0
    if need_gb is None:
        need_gb = float(needs.get(stage, spec.get("default_free_gb", 0.0)))
    return min(float(need_gb) * 1024, total_mib - float(spec.get("reserve_gb", 0.0)) * 1024)


def light_blocked(
    light: dict, stage: str, status: dict | None, need_mib: float, now: float | None = None
) -> str | None:
    """Why a light card (``spec["light"][gpu]``) takes no lease for ``stage`` now, or None."""
    if stage not in light.get("stages", ()):
        return "light_stage"
    guard = light.get("guard")
    if guard:
        try:
            state = json.loads(open(guard).read())
        except (OSError, ValueError):
            return "light_guard_missing"
        age = (now or time.time()) - float(state.get("at", 0.0))
        if age > float(light.get("guard_max_age_s", 180.0)):
            return "light_guard_stale"
        if not state.get("ok"):
            return "light_guard_vlm"
    if status is None:
        return "light_no_probe"  # never load a light card blind
    if gpu_hot(status, float(light.get("hot_c", 80.0))):
        return "light_hot"
    free_mib = status["mem_total_mib"] - status["mem_used_mib"]
    if free_mib - float(light.get("reserve_gb", 4.0)) * 1024 < need_mib:
        return "light_full"
    return None


def waiters(directory: Path | str) -> int:
    """Processes waiting for a lease in ``directory`` now (their marker locks are held)."""
    folder = Path(directory) / "waiting"
    count = 0
    for path in folder.glob("*.lock") if folder.is_dir() else ():
        handle = _try_lock(path)
        if handle is None:
            count += 1
            continue
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
        with contextlib.suppress(OSError):
            path.unlink()  # its waiter is gone
    return count


def free_units(spec: dict) -> dict[str, int]:
    """Unheld units per ordinary (not light) card of ``spec`` now."""
    directory = Path(spec["dir"])
    directory.mkdir(parents=True, exist_ok=True)
    light = spec.get("light") or {}
    return {
        str(g): int(n) - held_units(directory, str(g), int(n))
        for g, n in spec["gpus"].items()
        if str(g) not in light
    }


def try_acquire(
    spec: dict,
    stage: str,
    probe: Callable[[], dict[int, dict]] = probe_gpus,
    need_gb: float | None = None,
) -> tuple[Lease | None, str]:
    """One attempt: a :class:`Lease` or ``(None, reason)``."""
    if spec.get("gate") and not os.path.exists(spec["gate"]):
        return None, "cohort_wait"  # an older cohort's GPU work goes first (queue_cohorts)
    directory = Path(spec["dir"])
    directory.mkdir(parents=True, exist_ok=True)
    capacity = {str(g): int(n) for g, n in spec["gpus"].items()}
    want = int(spec.get("stage_units", {}).get(stage, spec.get("default_units", 1)))
    held = {g: held_units(directory, g, n) for g, n in capacity.items()}
    free = {g: capacity[g] - held[g] for g in capacity}
    fits = [g for g in capacity if free[g] >= min(want, capacity[g])]
    if not fits:
        return None, "no_free_units"
    status = probe() or {}
    hot_c = float(spec.get("hot_c", 83.0))
    per_unit_mib = float(spec.get("min_free_gb_per_unit", 0.0)) * 1024
    light = spec.get("light") or {}
    cool, hot, lights, light_reasons = [], [], [], []
    for g in fits:
        s = status.get(int(g))
        if g in light:
            need = max(
                per_unit_mib * min(want, capacity[g]),
                stage_need_mib(spec, stage, s["mem_total_mib"], need_gb) if s else 0.0,
            )
            why = light_blocked(light[g], stage, s, need)
            if why is None:
                lights.append((held[g], s["util_percent"], s["temp_c"], g))
            else:
                light_reasons.append(why)
            continue
        if s is None:
            cool.append((held[g], 0, 0, g))
            continue
        free_mib = s["mem_total_mib"] - s["mem_used_mib"]
        if free_mib < per_unit_mib * min(want, capacity[g]):
            continue
        if free_mib < stage_need_mib(spec, stage, s["mem_total_mib"], need_gb):
            continue
        entry = (held[g], s["util_percent"], s["temp_c"], g)
        (hot if gpu_hot(s, hot_c) else cool).append(entry)
    ranked = sorted(cool) or sorted(e for e in hot if e[0] == 0) or sorted(lights)
    if not ranked:
        ordinary = [g for g in fits if g not in light]
        reasons = [r for r in light_reasons if r != "light_stage"]
        if ordinary or not light_reasons:
            return None, "gpu_hot_or_full"
        return None, reasons[0] if reasons else "no_free_units"
    for *_, g in ranked:
        n = min(want, capacity[g])
        handles = []
        for unit in range(capacity[g]):
            handle = _try_lock(_unit_path(directory, g, unit))
            if handle is not None:
                handles.append(handle)
                if len(handles) == n:
                    return Lease(g, n, handles), "ok"
        Lease(g, 0, handles).release()  # another process took units in between
    return None, "raced"


def acquire(
    spec: dict,
    stage: str,
    *,
    probe: Callable[[], dict[int, dict]] = probe_gpus,
    poll_seconds: float = 10.0,
    sleep: Callable[[float], None] = time.sleep,
    need_gb: float | None = None,
) -> tuple[Lease, dict]:
    """Block until a lease is granted; returns it with a record of the wait."""
    started = time.time()
    reasons: dict[str, int] = {}
    marker = None
    try:
        while True:
            lease, reason = try_acquire(spec, stage, probe, need_gb)
            if lease is not None:
                return lease, {"waited_seconds": round(time.time() - started, 1), "blocks": reasons}
            reasons[reason] = reasons.get(reason, 0) + 1
            if marker is None and reason != "cohort_wait":
                marker = _wait_marker(Path(spec["dir"]), stage)
            sleep(poll_seconds)
    finally:
        if marker is not None:
            path, handle = marker
            with contextlib.suppress(OSError):
                path.unlink()
            handle.close()


def _wait_marker(directory: Path, stage: str):
    """Hold ``<dir>/waiting/<pid>.<stage>.lock`` while waiting (released if the waiter dies)."""
    folder = directory / "waiting"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{os.getpid()}.{threading.get_ident()}.{stage}.lock"
        handle = _try_lock(path)
    except OSError:
        return None
    return (path, handle) if handle is not None else None


class _CardSampler:
    """Samples one card while a lease is held: peak temperature, lowest SM clock, throttling."""

    def __init__(self, gpu: int, every: float, probe: Callable[[], dict[int, dict]]) -> None:
        self.gpu, self.every, self.probe = gpu, every, probe
        self.peak: int | None = None
        self.min_sm: int | None = None
        self.throttled = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.every):
            s = (self.probe() or {}).get(self.gpu)
            if s:
                self.peak = max(self.peak or s["temp_c"], s["temp_c"])
                self.min_sm = min(self.min_sm or s["sm_mhz"], s["sm_mhz"])
                self.throttled |= bool(s["throttle"] & THERMAL_THROTTLE_BITS)

    def __enter__(self) -> "_CardSampler":
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()

    def record(self) -> dict:
        return {
            "temp_c_peak": self.peak,
            "sm_mhz_min": self.min_sm,
            "thermal_throttle": self.throttled,
        }


def _log(spec: dict, record: dict) -> None:
    path = os.environ.get(ENV_LOG) or spec.get("log")
    if not path:
        return
    with contextlib.suppress(OSError):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as handle:
            handle.write(json.dumps(record) + "\n")


@contextlib.contextmanager
def lease(
    stage: str,
    probe: Callable[[], dict[int, dict]] = probe_gpus,
    need_gb: float | None = None,
) -> Iterator[dict]:
    """Hold a GPU for ``stage``; yields the environment overrides for its processes.

    ``{}`` when leasing is off or this process tree already holds a lease. ``need_gb``
    overrides the spec's free-memory need for this stage (see :func:`stage_need_mib`).
    """
    spec = spec_from_env()
    if spec is None or os.environ.get(ENV_HELD):
        yield {}
        return
    held, wait = acquire(spec, stage, probe=probe, need_gb=need_gb)
    status = (probe() or {}).get(int(held.gpu), {})
    started = time.time()
    every = spec.get("sample_seconds")
    sampler = _CardSampler(int(held.gpu), float(every), probe) if every else None
    try:
        with sampler or contextlib.nullcontext():
            yield {"CUDA_VISIBLE_DEVICES": held.gpu, ENV_HELD: held.gpu}
    finally:
        end = (probe() or {}).get(int(held.gpu), {})
        held.release()
        _log(
            spec,
            {
                "stage": stage,
                "gpu": held.gpu,
                "units": held.units,
                "pid": os.getpid(),
                "started": round(started, 1),
                "held_seconds": round(time.time() - started, 1),
                **wait,
                **({"need_gb": need_gb} if need_gb is not None else {}),
                "temp_c_start": status.get("temp_c"),
                "temp_c_end": end.get("temp_c"),
                "sm_mhz_start": status.get("sm_mhz"),
                "sm_mhz_end": end.get("sm_mhz"),
                **(sampler.record() if sampler else {}),
                **({"light": True} if held.gpu in (spec.get("light") or {}) else {}),
            },
        )


@contextlib.contextmanager
def environment(stage: str) -> Iterator[None]:
    """Hold a lease and point this process's ``os.environ`` at it (single-threaded callers:
    stage commands and in-process callbacks that start GPU subprocesses)."""
    with lease(stage) as override:
        saved = {key: os.environ.get(key) for key in override}
        os.environ.update(override)
        try:
            yield
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def run(
    command: list[str], *, stage: str, need_gb: float | None = None, **kwargs
) -> subprocess.CompletedProcess:
    """``subprocess.run`` under a lease (thread-safe: passes ``env`` rather than editing
    ``os.environ``)."""
    with lease(stage, need_gb=need_gb) as override:
        if override:
            kwargs["env"] = {**(kwargs.get("env") or os.environ), **override}
        return subprocess.run(command, **kwargs)
