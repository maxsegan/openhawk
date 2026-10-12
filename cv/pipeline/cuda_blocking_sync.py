"""Make CUDA waits sleep instead of spin (throughput only; results unchanged).

By default a host thread waiting for the GPU (``.cpu()``, ``synchronize``, a blocking copy)
spins a full core until the kernels finish. The pipeline's GPU stages wait most of the time
(four stage-1 model threads, two crop-refine threads, ...), so on a CPU-bound box every waiting
thread costs a core. Setting the primary context's scheduling flag to blocking sync makes those
waits sleep (measured: 0.76 -> 0.01 CPU-s per 0.77 s of waiting, same wall).

The flag must be set before the process creates its CUDA context, so callers invoke
:func:`enable` at the very start of ``main`` (before any torch CUDA call). It is a no-op unless
``TENNIS_CUDA_BLOCKING_SYNC=1``, and it never fails a stage: a missing driver or an
already-active context leaves the default behaviour and says so on stderr.
"""

from __future__ import annotations

import ctypes
import os
import sys

CU_CTX_SCHED_BLOCKING_SYNC = 0x04


def enable() -> bool:
    """Set blocking sync on every visible device's primary context; True when applied."""
    if os.environ.get("TENNIS_CUDA_BLOCKING_SYNC", "0") != "1":
        return False
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        if cuda.cuInit(0) != 0:
            raise OSError("cuInit failed")
        count = ctypes.c_int()
        if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
            raise OSError("cuDeviceGetCount failed")
        for ordinal in range(count.value):
            device = ctypes.c_int()
            if cuda.cuDeviceGet(ctypes.byref(device), ordinal) != 0:
                raise OSError(f"cuDeviceGet({ordinal}) failed")
            status = cuda.cuDevicePrimaryCtxSetFlags(device, CU_CTX_SCHED_BLOCKING_SYNC)
            if status != 0:
                raise OSError(f"cuDevicePrimaryCtxSetFlags({ordinal}) returned {status}")
    except OSError as error:
        from cv.pipeline import switch_fallbacks

        print(f"cuda_blocking_sync: left default scheduling ({error})", file=sys.stderr)
        switch_fallbacks.record("TENNIS_CUDA_BLOCKING_SYNC", "default_scheduling", str(error))
        return False
    return True
