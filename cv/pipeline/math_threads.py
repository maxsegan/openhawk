"""Math-library thread pools for pipeline processes (``TENNIS_MATH_THREADS``).

Unset (default, rollback): every library sizes its own pool. On the 64-thread local box
OpenBLAS then starts 64 threads per process (numpy and scipy each load a copy) that spin
between calls: ``import numpy`` costs 2.9 CPU-s instead of 0.07, ``import scipy.optimize``
5.9, and a loop of 200x200 matrix products 9x the CPU for 2x the speed. Only the S6 fit
subprocesses limited themselves (one thread).

``TENNIS_MATH_THREADS=N``: OpenBLAS, OpenMP, MKL, numexpr and OpenCV pools get N threads in
the process that applies it (``apply``, through threadpoolctl for libraries already loaded)
and in every child it starts (the variables are set in ``os.environ``). The queue sets them
in each step's environment (``environment``), so a step starts already limited. A variable a
caller sets explicitly for its own child (the S6 fit subprocesses' one thread) still wins
there. The switch itself is a ``TENNIS_*`` variable, so step markers record it.
"""

from __future__ import annotations

import os

SWITCH = "TENNIS_MATH_THREADS"
VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",
)


def threads(environ=None) -> int | None:
    """The switch's thread count, or None when unset (libraries size their own pools)."""
    value = (os.environ if environ is None else environ).get(SWITCH, "").strip()
    if value in ("", "off"):
        return None
    count = int(value)
    if count < 1:
        raise ValueError(f"{SWITCH} must be a positive thread count or off")
    return count


def environment(environ=None) -> dict[str, str]:
    """The variables a child process needs for the switch (empty when it is unset)."""
    count = threads(environ)
    if count is None:
        return {}
    return {SWITCH: str(count), **{name: str(count) for name in VARIABLES}}


def apply() -> int | None:
    """Limit this process (libraries already loaded) and every child it starts."""
    count = threads()
    if count is None:
        return None
    os.environ.update(environment())
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - threadpoolctl ships with the cv extra
        pass
    else:
        threadpool_limits(limits=count)
    try:
        import cv2
    except ImportError:  # pragma: no cover
        pass
    else:
        cv2.setNumThreads(count)
    return count
