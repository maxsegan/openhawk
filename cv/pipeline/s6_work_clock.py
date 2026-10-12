"""S6 fit budgets counted in work done instead of wall time (owner addendum, 2026-10-01).

The S6 fit's budgets (the 900 s search deadline, the refinement stage caps, the component
orchestrator's remaining-time shares) compare ``time.monotonic()`` against a limit, so on a
loaded box the same inputs can stop earlier and give a different fit. With
``TENNIS_S6_WORK_BUDGET=on`` those comparisons read this clock instead: a process-wide
counter that advances by one unit on every budget read and by a size-proportional amount on
every GPU solve, scaled to seconds by ``UNIT_SECONDS`` (calibrated so that a typical attempt
spends about as many clock-seconds as it took on an unloaded box). The same inputs and code
then make the same reads in the same order, whatever the load, so budgets bind identically.
On by default (owner addendum, validated in cv/experiments/fit_lean/FINAL_REPORT_fit_lean.md §7).

Wall time is still measured and reported where it was; only budget decisions move. The
process timeouts (subprocess kill at the attempt timeout) stay real-time safety nets.
Rollback: ``TENNIS_S6_WORK_BUDGET=off`` (wall-clock budgets, as before).
"""

from __future__ import annotations

import os
import threading
import time

SWITCH = "TENNIS_S6_WORK_BUDGET"
# Clock-seconds per budget read, per fit path (fixed by configuration, so the clock stays
# deterministic): the GPU fit ~5.6 ms plus ~2.8 us per GPU objective evaluation (fit-lean §7);
# the CPU fit below (gpu_fit FINAL_REPORT §13).
_GPU_PATH = os.environ.get("TENNIS_S6_GPU_FIT", "off") == "on"
UNIT_SECONDS = float(
    os.environ.get("TENNIS_S6_WORK_UNIT_SECONDS", "0.0056" if _GPU_PATH else "0.0004")
)
# CPU fit (Addendum 7, 2026-10-02): a budget read inside ``real_exposure_replay.refine`` (its
# SLSQP constraint evaluations) costs 0.47 / 0.92 / 2.94 ms of CPU (p10/p50/p90, 1,177 refine
# segments of 48 fresh first fits), growing with the problem: about 0.71 ms + 0.21 ms per flight
# + 0.0011 ms per simulated frame. The old flat 3.0 ms ran the clock a median 3.3x ahead of the
# CPU, so budgets cut fits the wall clock let finish. A refine read is now charged that size
# model at 0.40 of its fit (ahead of the CPU in 3.9% of segments, 0.5% of CPU time); other
# reads cost UNIT_SECONDS (0.4 ms, below the 10th-percentile read). Deterministic as before.
CPU_READ_SECONDS = tuple(
    float(v)
    for v in os.environ.get("TENNIS_S6_WORK_CPU_READ", "0.000285,0.0000845,0.00000044").split(",")
)


def refine_read_units(flights: int, span_frames: float) -> float:
    """Budget-read units one refine read stands for (1 on the GPU path, calibrated there)."""
    if _GPU_PATH:
        return 1.0
    base, per_flight, per_frame = CPU_READ_SECONDS
    return max(1.0, (base + per_flight * flights + per_frame * span_frames) / UNIT_SECONDS)


# Budget-read units charged per GPU objective evaluation (search, finish or gradient launch).
GPU_EVALUATION_UNITS = float(os.environ.get("TENNIS_S6_WORK_GPU_UNITS", "0.0005"))
LOG = "TENNIS_S6_WORK_CLOCK_LOG"

_lock = threading.Lock()
_units = 0
_log = None


def enabled() -> bool:
    value = os.environ.get(SWITCH, "on")
    if value not in ("on", "off", "1", "0"):
        raise ValueError(f"{SWITCH} must be on or off")
    return value in ("on", "1")


def _record(site: str, units: float) -> None:
    global _log
    if _log is None:
        path = os.environ[LOG]
        os.makedirs(path, exist_ok=True)
        _log = open(os.path.join(path, f"{os.getpid()}.tsv"), "a", buffering=1 << 16)
    _log.write(f"{site}\t{units}\t{time.monotonic():.6f}\t{time.process_time():.6f}\n")


def now(site: str = "budget") -> float:
    """The budget clock: work clock-seconds with the switch on, else ``time.monotonic()``."""
    global _units
    if os.environ.get(LOG):
        _record(site, 1)
    if not enabled():
        return time.monotonic()
    with _lock:
        _units += 1
        return _units * UNIT_SECONDS


def advance(units: float, site: str = "gpu") -> None:
    """Charge work done outside budget reads (a GPU solve), in budget-read units."""
    global _units
    if os.environ.get(LOG):
        _record(site, units)
    if enabled():
        with _lock:
            _units += units


def units() -> float:
    return _units


def kill_factor() -> float:
    """Real-time process kills stay safety nets: generous when budgets count work."""
    return float(os.environ.get("TENNIS_S6_WORK_KILL_FACTOR", "4")) if enabled() else 1.0


def receipt() -> dict | None:
    """This process's clock for its result (the parent adds ``units`` to its own clock)."""
    if not enabled():
        return None
    return {
        "schema": "s6_work_clock_v1",
        "units": _units,
        "unit_seconds": UNIT_SECONDS,
        "gpu_evaluation_units": GPU_EVALUATION_UNITS,
        "clock_seconds": _units * UNIT_SECONDS,
    }
