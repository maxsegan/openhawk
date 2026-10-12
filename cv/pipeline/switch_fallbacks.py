"""Loud, recorded fallbacks of the throughput/GPU switches (one record per process and reason).

Every candidate switch has a slower path the stage takes when the box lacks the capability
(TensorRT bindings or engines, PyNvVideoCodec/NVDEC, the court CUDA kernels without nvcc,
nvJPEG, the GPU fit kernel). A fallback keeps the stage running, but the match then ran a
different configuration from the one the queue asked for, so it must never be silent:

- :func:`record` warns on stderr and, when ``TENNIS_FALLBACK_LOG`` names a file, appends one
  JSON line (switch, stable ``kind``, detail, where, host, pid). The product runner points it at
  ``logs/fallbacks_<step>.jsonl`` for each step, so stage subprocesses on any box write there;
- :func:`summary` is what the step marker, ``OUTCOMES.json`` and the queue receipt carry;
- :func:`probe` (``python -m cv.pipeline.switch_fallbacks --probe``) is the box check: for each
  switch that is on, whether this box can honour it and, if not, which fallback it will take;
- :func:`snapshot` is the switch record every step marker carries: the ``TENNIS_*`` values in
  effect (never a name containing KEY, TOKEN, SECRET or PASSWORD), the runtime pin and, on a
  remote box step, the box's effective environment; ``product_reprocess rerun`` replays it
  (:func:`apply`, without the locations in :data:`NOT_REPLAYED`).

A fallback repeated for every frame or call is recorded once per process for each
(switch, kind, where), so a log line means "at least once in that process", not a count.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
from pathlib import Path

ENV = "TENNIS_FALLBACK_LOG"
#: Set by product_boxes on a box whose GPU fit is declined (kernel did not load, single card):
#: the reason, recorded by the runner at the start of every step that runs there.
GPU_FIT_DECLINED_ENV = "TENNIS_S6_GPU_FIT_DECLINED"

#: Switches whose fallback gives the same bytes (throughput only). Every other fallback may
#: change outputs (TensorRT -> torch fp16, CUDA SIFT -> the port, GPU court support -> CPU
#: distances, GPU fit -> CPU solves) and is flagged as such.
OUTPUT_NEUTRAL = frozenset(
    {
        "TENNIS_VIDEO_DECODE",
        "TENNIS_COURT_DECODE",
        "TENNIS_EVENT_CROPS_DECODE",
        "TENNIS_CUDA_BLOCKING_SYNC",
    }
)

_seen: set[tuple[str, str, str]] = set()
_lock = threading.Lock()


def record(switch: str, kind: str, detail: str = "", where: str = "") -> bool:
    """Report one fallback of ``switch`` (first time per process for this kind and place).

    ``kind`` is a stable short code (``module_missing``, ``kernel_unavailable``,
    ``low_memory``, ...); ``detail`` the error text. Returns True when it was written.
    """
    key = (switch, kind, where)
    with _lock:
        if key in _seen:
            return False
        _seen.add(key)
    detail = str(detail)[:400]
    print(
        f"WARNING fallback: {switch} {kind}" + (f" in {where}" if where else "") + f": {detail}",
        file=sys.stderr,
        flush=True,
    )
    path = os.environ.get(ENV)
    if path:
        line = {
            "switch": switch,
            "value": os.environ.get(switch),
            "kind": kind,
            "where": where,
            "detail": detail,
            "host": socket.gethostname(),
            "pid": os.getpid(),
        }
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as handle:  # one short line per write: appends do not interleave
                handle.write(json.dumps(line) + "\n")
        except OSError as error:
            print(f"WARNING fallback: cannot write {path}: {error}", file=sys.stderr, flush=True)
    return True


def reset() -> None:
    """Forget what this process recorded (a new step logs its own first occurrences)."""
    with _lock:
        _seen.clear()


def error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def read(path: Path) -> list[dict]:
    rows = []
    try:
        with open(path) as handle:
            for line in handle:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return rows


def summary(path: Path) -> dict | None:
    """``{switch: {kind: {"processes": n, "where": [...], "example": detail}}}``, or None."""
    out: dict[str, dict] = {}
    for row in read(path):
        entry = out.setdefault(row.get("switch", "?"), {}).setdefault(
            row.get("kind", "?"), {"processes": 0, "where": [], "example": row.get("detail")}
        )
        entry["processes"] += 1
        if row.get("where") and row["where"] not in entry["where"]:
            entry["where"].append(row["where"])
    return out or None


def merge(summaries: dict[str, dict | None]) -> dict:
    """Per-step summaries to ``{switch: sorted kinds}`` (the receipt's one-line view)."""
    merged: dict[str, set] = {}
    for found in summaries.values():
        for switch, kinds in (found or {}).items():
            merged.setdefault(switch, set()).update(kinds)
    return {switch: sorted(kinds) for switch, kinds in sorted(merged.items())}


# --------------------------------------------------------------------------- switch record

#: Names never recorded: anything whose name contains one of these (credentials).
SECRET_PARTS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
#: Set by product_queue on a remote box step: the box name and its ``Box.environment()``.
BOX_ENVIRONMENT_ENV = "PRODUCT_BOX_ENVIRONMENT"
#: Recorded but never replayed by a rerun: locations, logs and per-box plumbing (the rerun
#: has its own data root, logs and fit route; a replayed decline would record a false fallback).
NOT_REPLAYED = frozenset(
    {
        "TENNIS_DATA_ROOT",
        "TENNIS_DATA_ROOT_ORIGINAL",
        "TENNIS_TRACKER_ROOT",
        "TENNIS_TRT_DIR",
        "TENNIS_TRT_ENGINE_DIR",
        "TENNIS_S6_GPU_FIT_SERVER",
        "TENNIS_S6_WORK_CLOCK_LOG",
        "TENNIS_POSE_GPU_LOCK_DIR",
        ENV,
        GPU_FIT_DECLINED_ENV,
    }
)
_RUNTIME: dict | None = None


def secret(name: str) -> bool:
    return any(part in name.upper() for part in SECRET_PARTS)


def values(environ=None) -> dict[str, str]:
    """Every ``TENNIS_*`` switch in ``environ`` (default: this process), secrets excluded."""
    environ = os.environ if environ is None else environ
    return {k: v for k, v in sorted(environ.items()) if k.startswith("TENNIS_") and not secret(k)}


def replayable(recorded: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in recorded.items() if k not in NOT_REPLAYED and not secret(k)}


def runtime() -> dict:
    """The checkout this process runs from (the runtime pin): path, commit, dirty."""
    global _RUNTIME
    if _RUNTIME is None:
        from cv.pipeline import provenance

        root = Path(__file__).resolve().parents[2]
        _RUNTIME = {"path": str(root), **provenance.git_record(root)}
    return dict(_RUNTIME)


def snapshot() -> dict:
    """What a step marker records: the switch values in effect, the memset tunable, the host,
    the runtime pin and, on a remote box step, the box's effective environment."""
    record = {
        "values": values(),
        "glibc_tunables": os.environ.get("GLIBC_TUNABLES"),
        "host": socket.gethostname(),
        "runtime": runtime(),
    }
    box = os.environ.get(BOX_ENVIRONMENT_ENV)
    if box:
        try:
            record["box"] = json.loads(box)
        except json.JSONDecodeError:
            record["box"] = {"unparsed": box[:2000]}
    return record


def box_environment(name: str, environment: dict[str, str]) -> str:
    """``PRODUCT_BOX_ENVIRONMENT`` for a box step (secrets excluded)."""
    kept = {k: v for k, v in sorted(environment.items()) if not secret(k)}
    return json.dumps({"box": name, "environment": kept}, sort_keys=True)


def apply(switches: dict[str, str], environ=None) -> dict[str, str | None]:
    """Make ``switches`` the replayable ``TENNIS_*`` switches of ``environ`` (others removed);
    returns what to pass to :func:`restore`."""
    environ = os.environ if environ is None else environ
    wanted = replayable(switches)
    names = {k for k in values(environ) if k not in NOT_REPLAYED} | set(wanted)
    previous = {k: environ.get(k) for k in names}
    for name in names:
        if name in wanted:
            environ[name] = wanted[name]
        else:
            environ.pop(name, None)
    return previous


def restore(previous: dict[str, str | None], environ=None) -> None:
    environ = os.environ if environ is None else environ
    for name, value in previous.items():
        if value is None:
            environ.pop(name, None)
        else:
            environ[name] = value


def flags(switches) -> list[str]:
    """Receipt flags for the switches that fell back: ``switch_fallbacks`` when outputs may
    differ from the configuration asked for, ``switch_fallbacks_throughput`` when every one
    of them gives the same bytes."""
    switches = set(switches)
    if switches - OUTPUT_NEUTRAL:
        return ["switch_fallbacks"]
    return ["switch_fallbacks_throughput"] if switches else []


# --------------------------------------------------------------------------- box probe


def _on(name: str, *values: str) -> bool:
    return os.environ.get(name, "") in values


def _probe_tensorrt(devices: list[int]) -> dict:
    from cv.pipeline import ball_trt

    try:
        trt = ball_trt.tensorrt()
    except ImportError as error:
        return {
            "ok": False,
            "fallback": "torch_fp16 trackers (different numerics from the engines)",
            "reason": error_text(error),
        }
    engines = sorted(p.name for p in ball_trt.engine_dir().glob("*.engine"))
    return {
        "ok": True,
        "tensorrt": trt.__version__,
        "engines": len(engines),
        "note": None if engines else "no engines yet: the first tracking step builds them",
        "devices": devices,
    }


def _probe_nvdec(device: int) -> dict:
    try:
        import PyNvVideoCodec  # noqa: F401
    except ImportError as error:
        return {
            "ok": False,
            "fallback": "CPU ffmpeg decode (same bytes)",
            "reason": error_text(error),
        }
    return {"ok": True}


def _probe_court_kernel(module: str, fallback: str) -> dict:
    import importlib

    try:
        importlib.import_module(f"cv.experiments.gpu_court.{module}").library()
    except Exception as error:  # noqa: BLE001 - no nvcc and no prebuilt kernel, or no CUDA
        return {"ok": False, "fallback": fallback, "reason": error_text(error)[:400]}
    return {"ok": True}


def _probe_nvjpeg(device: int) -> dict:
    try:
        from cv.experiments.gpu_decode import gpu_jpeg

        decoder = gpu_jpeg.NvJpegDecoder(device)
        del decoder
    except Exception as error:  # noqa: BLE001 - no library, or the card has no NVJPG engine
        return {"ok": False, "fallback": "cv2.imread (same pictures)", "reason": error_text(error)}
    return {"ok": True}


def probe(device: int = 0) -> dict:
    """What this process's switches need, checked on CUDA ``device``: ``{switch: result}``.

    ``ok`` False means the stage will take ``fallback`` (and record it when it does).
    """
    report: dict[str, dict] = {}
    if os.environ.get("TENNIS_BALL_ENGINE", "torch") not in ("", "torch"):
        report["TENNIS_BALL_ENGINE"] = _probe_tensorrt([device])
    if os.environ.get("TENNIS_PLAYER_ENGINE", "torch") not in ("", "torch"):
        report["TENNIS_PLAYER_ENGINE"] = _probe_tensorrt([device])
        if not report["TENNIS_PLAYER_ENGINE"]["ok"]:
            report["TENNIS_PLAYER_ENGINE"]["fallback"] = "torch fp32 player detection (default)"
    if os.environ.get("TENNIS_VIDEO_DECODE", "cpu").startswith("nvdec"):
        report["TENNIS_VIDEO_DECODE"] = _probe_nvdec(device)
    sift = os.environ.get("TENNIS_COURT_SIFT", "cpu")
    if sift == "cuda":
        report["TENNIS_COURT_SIFT"] = _probe_court_kernel(
            "sift_cuda", "PyTorch SIFT port (TENNIS_COURT_SIFT=gpu numerics)"
        )
    if _on("TENNIS_COURT_SUPPORT", "gpu"):
        report["TENNIS_COURT_SUPPORT"] = _probe_court_kernel(
            "dt_cuda", "CPU line masks and distances"
        )
    for switch in ("TENNIS_COURT_DECODE", "TENNIS_EVENT_CROPS_DECODE"):
        if _on(switch, "nvjpeg"):
            report[switch] = _probe_nvjpeg(device)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--probe", action="store_true", help="print the box's switch report")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--summary", type=Path, help="summarise a fallbacks_<step>.jsonl")
    args = parser.parse_args()
    if args.summary:
        print(json.dumps(summary(args.summary), indent=1))
    else:
        print(json.dumps(probe(args.device)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
