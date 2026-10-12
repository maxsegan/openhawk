"""Bit-identical scalar rewrite of the connected-search propagation inner loop.

Question this answers: the whole-point search spends nearly all of its time in
``physics.flight.rk4_step`` at 240 Hz, where every three-vector operation pays a
NumPy dispatch.  This module performs exactly the same double-precision
arithmetic, in exactly the same order, on Python floats.  ``test_fast_flight.py``
asserts bit-for-bit equality against the shared physics module over randomized
states, so replacing one with the other cannot change a fitted family.

It is an optimization of an existing model, not a new model: no coefficient,
step size, ordering or tolerance is changed here.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
from pathlib import Path
import subprocess

import numpy as np

from physics import flight


_P = flight.default_params
G = _P["g"]
K = 0.5 * _P["rho_air"] * np.pi * _P["R_ball"] ** 2 / _P["m_ball"]
K4 = 0.5 * _P["rho_air"] * np.pi * _P["R_ball"] ** 4 / _P["J_ball"]
C_DRAG = _P["C_drag"]
LIFT_SCALE = K * _P["C_lift"] * _P["R_ball"]
SPIN_DECAY = -_P.get("C_spin_decay", 0.025) * K4


def derivatives(vx, vy, vz, wx, wy, wz):
    """Return ``(accel, w_dot)`` components; same order of operations as physics."""
    vmag = math.sqrt(vx * vx + vy * vy + vz * vz)
    wmag = math.sqrt(wx * wx + wy * wy + wz * wz)
    ax = 0.0
    ay = 0.0
    az = -G
    if vmag > 1e-12:
        drag_scale = -K * C_DRAG * vmag
        ax += drag_scale * vx
        ay += drag_scale * vy
        az += drag_scale * vz
        if wmag > 1e-12:
            ax += LIFT_SCALE * (wy * vz - wz * vy)
            ay += LIFT_SCALE * (wz * vx - wx * vz)
            az += LIFT_SCALE * (wx * vy - wy * vx)
    spin_scale = SPIN_DECAY * vmag
    return ax, ay, az, spin_scale * wx, spin_scale * wy, spin_scale * wz


def rk4_step(state, dt):
    """Advance ``(x, v, w)`` packed as nine floats by one RK4 step."""
    px, py, pz, vx, vy, vz, wx, wy, wz = state
    a1x, a1y, a1z, s1x, s1y, s1z = derivatives(vx, vy, vz, wx, wy, wz)

    m1vx = vx + a1x * dt / 2
    m1vy = vy + a1y * dt / 2
    m1vz = vz + a1z * dt / 2
    m1wx = wx + s1x * dt / 2
    m1wy = wy + s1y * dt / 2
    m1wz = wz + s1z * dt / 2
    a2x, a2y, a2z, s2x, s2y, s2z = derivatives(m1vx, m1vy, m1vz, m1wx, m1wy, m1wz)

    m2vx = vx + a2x * dt / 2
    m2vy = vy + a2y * dt / 2
    m2vz = vz + a2z * dt / 2
    m2wx = wx + s2x * dt / 2
    m2wy = wy + s2y * dt / 2
    m2wz = wz + s2z * dt / 2
    a3x, a3y, a3z, s3x, s3y, s3z = derivatives(m2vx, m2vy, m2vz, m2wx, m2wy, m2wz)

    evx = vx + a3x * dt
    evy = vy + a3y * dt
    evz = vz + a3z * dt
    ewx = wx + s3x * dt
    ewy = wy + s3y * dt
    ewz = wz + s3z * dt
    a4x, a4y, a4z, s4x, s4y, s4z = derivatives(evx, evy, evz, ewx, ewy, ewz)

    return (
        px + dt * (vx + 2 * m1vx + 2 * m2vx + evx) / 6,
        py + dt * (vy + 2 * m1vy + 2 * m2vy + evy) / 6,
        pz + dt * (vz + 2 * m1vz + 2 * m2vz + evz) / 6,
        vx + dt * (a1x + 2 * a2x + 2 * a3x + a4x) / 6,
        vy + dt * (a1y + 2 * a2y + 2 * a3y + a4y) / 6,
        vz + dt * (a1z + 2 * a2z + 2 * a3z + a4z) / 6,
        wx + dt * (s1x + 2 * s2x + 2 * s3x + s4x) / 6,
        wy + dt * (s1y + 2 * s2y + 2 * s3y + s4y) / 6,
        wz + dt * (s1z + 2 * s2z + 2 * s3z + s4z) / 6,
    )


def exceeds(px, py, pz, limit):
    """Match ``np.linalg.norm(...) > limit`` exactly at a fraction of the cost.

    The scalar sum can differ from NumPy's reduction in the last bit, so the
    rare near-threshold comparison falls back to the array reduction and the
    decision is provably the one the previous code took.
    """
    value = math.sqrt(px * px + py * py + pz * pz)
    if abs(value - limit) <= 1e-9 * limit:
        value = float(np.linalg.norm(np.array((px, py, pz))))
    return value > limit


# --------------------------------------------------------------------------- native loop
#
# ``fast_flight_native.c`` runs the ordinary free-flight steps of
# ``measured_dynamics.integrate`` with this module's arithmetic (see that file). It is compiled
# on first use with the system C compiler into a per-user cache; without a compiler, or with
# ``TENNIS_FAST_FLIGHT=python``, the Python loop runs instead. Both give identical traces
# (``test_fast_flight_native.py``), so the switch is a speed rollback, not a model choice.

FF_END, FF_STOP, FF_CAPACITY, FF_IMPACT = 0, 1, 2, 3
_NATIVE_SOURCE = Path(__file__).with_name("fast_flight_native.c")
_NATIVE_FLAGS = ("-O2", "-shared", "-fPIC", "-ffp-contract=off", "-fno-fast-math", "-lm")
_CONSTS = np.array([G, K, C_DRAG, LIFT_SCALE, SPIN_DECAY], dtype=np.float64)
_CONSTS_ADDRESS = _CONSTS.ctypes.data
_native_library = None
_native_checked = False


def _compile_native():
    source = _NATIVE_SOURCE.read_bytes()
    compiler = os.environ.get("CC", "cc")
    digest = hashlib.sha256(source + " ".join((compiler, *_NATIVE_FLAGS)).encode()).hexdigest()
    cache = Path(
        os.environ.get("TENNIS_NATIVE_CACHE", Path.home() / ".cache" / "tennis" / "native")
    )
    target = cache / f"fast_flight_{digest[:16]}.so"
    if not target.is_file():
        cache.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(f".{os.getpid()}.tmp")
        subprocess.run(
            [compiler, *_NATIVE_FLAGS[:-1], str(_NATIVE_SOURCE), "-o", str(partial), "-lm"],
            check=True,
            capture_output=True,
        )
        os.replace(partial, target)
    library = ctypes.CDLL(str(target))
    pointer = ctypes.c_void_p
    library.ff_advance.restype = ctypes.c_int
    library.ff_advance.argtypes = (
        pointer,
        pointer,
        pointer,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_long,
        pointer,
        pointer,
        pointer,
    )
    library.ff_advance_net.restype = ctypes.c_int
    library.ff_advance_net.argtypes = (
        pointer,
        pointer,
        pointer,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_long,
        pointer,
        pointer,
        pointer,
    )
    library.ff_first_impact.restype = ctypes.c_int
    library.ff_first_impact.argtypes = (
        pointer,
        pointer,
        pointer,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        pointer,
        pointer,
    )
    return library


_PYTHON_KERNEL = (derivatives, rk4_step, exceeds)


def native():
    """The compiled loop, or None (no compiler, or ``TENNIS_FAST_FLIGHT=python``).

    Also None while this module's Python kernel or constants are replaced (tests substitute
    simplified dynamics), since the compiled loop carries its own copy of both.
    """
    global _native_library, _native_checked
    if (derivatives, rk4_step, exceeds) != _PYTHON_KERNEL or (
        G,
        K,
        C_DRAG,
        LIFT_SCALE,
        SPIN_DECAY,
    ) != tuple(_CONSTS.tolist()):
        return None
    if not _native_checked:
        _native_checked = True
        if os.environ.get("TENNIS_FAST_FLIGHT", "native") != "python":
            try:
                _native_library = _compile_native()
            except (OSError, subprocess.SubprocessError):
                _native_library = None
    return _native_library


_NO_STEPS = (np.empty(0), np.empty((0, 9)))


def native_advance(library, state, elapsed, end, step, r_ball, net_y=None, store=True):
    """Ordinary steps from ``state`` at ``elapsed``: (status, state, elapsed, times, states).

    ``times`` (n,) and ``states`` (n, 9) are the taken steps; on ``FF_STOP`` the returned
    state and elapsed are the start of the step Python must take itself. With ``net_y`` it
    also stops before a step whose chord crosses the plane ``y = net_y``. ``store=False``
    takes the same steps without keeping them (empty ``times``/``states``).
    """
    current = np.array(state, dtype=np.float64)
    clock = np.array((elapsed,), dtype=np.float64)
    count = np.zeros(1, dtype=np.int64)  # C long
    if store:
        capacity = max(1, min(int((end - elapsed) / step) + 2, 1_000_000))
        times = np.empty(capacity, dtype=np.float64)
        states = np.empty((capacity, 9), dtype=np.float64)
        outputs = (capacity, times.ctypes.data, states.ctypes.data, count.ctypes.data)
    else:
        times, states = _NO_STEPS
        outputs = (0, None, None, count.ctypes.data)
    start = (_CONSTS_ADDRESS, current.ctypes.data, clock.ctypes.data, end, step, r_ball)
    if net_y is None:
        status = library.ff_advance(*start, *outputs)
    else:
        status = library.ff_advance_net(*start, float(net_y), *outputs)
    n = int(count[0])
    return status, tuple(current.tolist()), float(clock[0]), times[:n], states[:n]


def native_first_impact(library, state, elapsed, end, step, r_ball):
    """First descending ground crossing from ``state``: (status, elapsed, seconds, impact).

    ``FF_IMPACT``: the crossing lies ``seconds`` into the step that starts at ``elapsed`` and
    ``impact`` is the state there, exactly as ``ground_root.descending_ground_root`` returns it.
    ``FF_END``: no crossing step before ``end``. ``FF_STOP``: the Python integrator must decide.
    """
    current = np.array(state, dtype=np.float64)
    clock = np.array((elapsed,), dtype=np.float64)
    out = np.zeros(10, dtype=np.float64)  # seconds, then the nine-float impact state
    status = library.ff_first_impact(
        _CONSTS_ADDRESS,
        current.ctypes.data,
        clock.ctypes.data,
        end,
        step,
        r_ball,
        out.ctypes.data,
        out.ctypes.data + 8,
    )
    return status, float(clock[0]), float(out[0]), tuple(out[1:].tolist())
