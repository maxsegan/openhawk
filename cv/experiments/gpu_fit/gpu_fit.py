"""Host side of the batched GPU S6 fit (``s6_fit_kernel.cu``); research prototype.

Question: can every seed of every ``real_exposure_replay.refine`` problem of an attempt be
solved in one GPU launch, and does the result agree with the CPU fitter?

* ``pack(record)`` turns one captured refine call (``capture/recorder.py``) into the kernel's
  ``Problem`` struct, or raises ``Unsupported`` naming the first feature the kernel lacks (the
  caller then keeps the CPU solve).
* ``probe(record)`` runs the unchanged CPU ``refine`` up to its SLSQP call and returns the CPU
  closures (objective residual, cost, constraints, x0, bounds) for parity checks and polish.
* ``solve(problems, seeds)`` / ``evaluate(problem, qs)`` launch the kernel through ctypes.

The kernel binary is built with nvcc (``$NVCC``) for the installed cards' compute capabilities
(``TENNIS_S6_GPU_FIT_ARCH``, e.g. ``sm_120``, overrides) into ``TENNIS_S6_GPU_FIT_BUILD`` (default
``$TENNIS_DATA_ROOT/processed/gpu-fit/build``), keyed by source and flags; never inside the
checkout or the shared venv. Concurrent processes share one compile (file lock); prebuild once per
box before a queue starts with ``python -m cv.experiments.gpu_fit.gpu_fit`` (prints the binary path).
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cv.pipeline import s6_work_clock as work_clock

# Device numbers follow nvidia-smi (PCI bus order) among the visible devices.


#: Every GPU fit failure the hooks answer on the CPU instead (out of memory, a lost device, an
#: unreachable fit server) is counted here, warned on stderr and, when this names a file,
#: appended to it as one JSON line, so the fit step can count fallbacks across its processes.
ERROR_LOG_ENV = "TENNIS_S6_GPU_FIT_ERROR_LOG"
_ERRORS: dict = {"cpu_fallbacks": 0, "examples": []}


def record_error(where: str, error: BaseException) -> None:
    """Count one GPU failure answered on the CPU; loud by design (results then mix paths)."""
    import json
    import sys

    message = f"{type(error).__name__}: {error}"[:300]
    _ERRORS["cpu_fallbacks"] += 1
    if len(_ERRORS["examples"]) < 3:
        _ERRORS["examples"].append({"where": where, "error": message})
    print(f"WARNING gpu_fit: {where} fell back to the CPU ({message})", file=sys.stderr, flush=True)
    from cv.pipeline import switch_fallbacks

    switch_fallbacks.record("TENNIS_S6_GPU_FIT", "cpu_fallback", message, where=where)
    path = os.environ.get(ERROR_LOG_ENV)
    if path:
        with open(path, "a") as handle:  # one short line per write: appends do not interleave
            handle.write(json.dumps({"pid": os.getpid(), "where": where, "error": message}) + "\n")


def error_receipt() -> dict:
    """This process's GPU-to-CPU fallbacks (for the attempt result)."""
    return {"cpu_fallbacks": _ERRORS["cpu_fallbacks"], "examples": list(_ERRORS["examples"])}


def logged_errors(path: Path) -> int:
    """Fallbacks recorded in a run's error log (0 when there were none)."""
    try:
        with open(path) as handle:
            return sum(1 for line in handle if line.strip())
    except FileNotFoundError:
        return 0


def default_device() -> int:
    """``TENNIS_S6_GPU_FIT_DEVICE`` (default 0): every launch without an explicit device."""
    return int(os.environ.get("TENNIS_S6_GPU_FIT_DEVICE", "0"))


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
SOURCE = Path(__file__).with_name("s6_fit_kernel.cu")
NVCC = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
MAX_F, MAX_OBS, MAX_P, MAX_WIN, MAX_EXP, MAX_ANCHOR, MAX_EP = 24, 1024, 152, 160, 3, 32, 8
NTHREADS = 64
ANCHOR_BOUNCE, ANCHOR_CONTACT, ANCHOR_CLOSING = 0, 1, 2
PARTIAL_ENDINGS = {"camera_cut", "held_camera", "point_end", "dead_ball"}


class Unsupported(ValueError):
    """The kernel does not model this refine configuration; keep the CPU solve."""


PROBLEM = np.dtype(
    [
        ("contact_frames", "f8", MAX_F + 1),
        ("obs_frames", "f8", MAX_OBS),
        ("cameras", "f8", MAX_OBS * 12),
        ("pixels", "f8", MAX_OBS * 2),
        ("radial", "f8", MAX_OBS * 3),
        ("exp_frames", "f8", MAX_F * MAX_EXP),
        ("scale", "f8", MAX_P),
        ("lo", "f8", MAX_P),
        ("hi", "f8", MAX_P),
        ("anchor_xy", "f8", MAX_ANCHOR * 2),
        ("anchor_sigma", "f8", MAX_ANCHOR * 2),
        ("ctarget", "f8", 3),
        ("ctarget_sigma", "f8", 3),
        ("ep_frame", "f8", MAX_EP),
        ("ep_sigma", "f8", MAX_EP),
        ("cue_root", "f8", 2),
        ("cue_axis", "f8", 2),
        ("cue_sigma", "f8"),
        ("cue_weight", "f8"),
        ("fps", "f8"),
        ("uncertainty", "f8"),
        ("dir_limit2", "f8"),
        ("last_frame", "f8"),
        ("term_margin0", "f8"),
        ("net_scale0", "f8"),
        ("net_scale", "f8"),
        ("surf", "f8", 8),
        ("phys", "f8", 8),
        ("net", "f8", 4),
        ("fixed_start", "f8", 3),
        ("fixed_scales", "f8", 2),
        ("init_target", "f8", 3),
        ("image_scale", "f8"),
        ("rebound_first", "f8"),
        ("net_frame", "f8", MAX_F),
        ("stop_target", "f8"),
        ("netk", "f8", 8),
        ("axes", "f8", MAX_OBS * 2),
        ("swept_duration", "f8"),
        ("swept_blur", "f8"),
        ("imaging_extra", "f8"),
        ("init_target_sigma", "f8"),
        ("n", "i4"),
        ("P", "i4"),
        ("nobs", "i4"),
        ("obs_off", "i4", MAX_F + 1),
        ("n_exp", "i4", MAX_F),
        ("nwin", "i4"),
        ("win_lo", "i4", MAX_WIN),
        ("win_hi", "i4", MAX_WIN),
        ("term_kind", "i4"),
        ("use_term_pen", "i4"),
        ("use_dir_pen", "i4"),
        ("has_radial", "i4"),
        ("n_anchor", "i4"),
        ("anchor_kind", "i4", MAX_ANCHOR),
        ("anchor_flight", "i4", MAX_ANCHOR),
        ("anchor_ordinal", "i4", MAX_ANCHOR),
        ("has_ctarget", "i4"),
        ("n_ep", "i4"),
        ("ep_flight", "i4", MAX_EP),
        ("ep_ordinal", "i4", MAX_EP),
        ("has_cue", "i4"),
        ("M", "i4"),
        ("M_objective", "i4"),
        ("pad", "i4"),
        ("kind", "i4"),
        ("soft_rows", "i4"),
        ("has_init_target", "i4"),
        ("has_rebound", "i4"),
        ("net_mode", "i4"),
        ("net_physical", "i4"),
        ("any_net", "i4"),
        ("has_stop_target", "i4"),
        ("pad5", "i4"),
        ("has_net", "i4", MAX_F),
        ("swept", "i4"),
        ("pad6", "i4"),
    ],
    align=True,
)
RESULT = np.dtype(
    [
        ("q", "f8", MAX_P),
        ("cost_initial", "f8"),
        ("cost_search", "f8"),
        ("cost_final", "f8"),
        ("objective_final", "f8"),
        ("iterations", "i4"),
        ("iterations64", "i4"),
        ("evaluations", "i4"),
        ("status", "i4"),
        ("cycles", "i8", 4),
    ],
    align=True,
)


class _Options(ctypes.Structure):
    _fields_ = [
        ("max_iter", ctypes.c_int),
        ("max_iter64", ctypes.c_int),
        ("m_stride", ctypes.c_int),
        ("p_stride", ctypes.c_int),
        ("ftol", ctypes.c_double),
        ("ftol64", ctypes.c_double),
        ("lambda0", ctypes.c_double),
        ("small_steps_to_stop", ctypes.c_int),
        ("central32", ctypes.c_int),
        ("report64", ctypes.c_int),
        ("pad", ctypes.c_int),
    ]


_LIBRARY = None


def build_dir() -> Path:
    if os.environ.get("TENNIS_S6_GPU_FIT_BUILD"):
        return Path(os.environ["TENNIS_S6_GPU_FIT_BUILD"])
    root = Path(os.environ.get("TENNIS_DATA_ROOT", "data"))
    return root / "processed" / "gpu-fit" / "build"


#: Every card we run on (3090/3080: sm_86; 5090 and RTX PRO 6000: sm_120), plus sm_120 PTX that
#: newer cards JIT. Fixed, not probed: every box then names the same library (same source, same
#: flags), so one build here is pushed to boxes without nvcc (``product_boxes setup``).
FLEET_ARCHITECTURES = ("sm_86", "sm_120")


def architectures() -> list[str]:
    """``sm_XY`` the library is built for (``TENNIS_S6_GPU_FIT_ARCH``, comma separated, overrides)."""
    if os.environ.get("TENNIS_S6_GPU_FIT_ARCH"):
        return os.environ["TENNIS_S6_GPU_FIT_ARCH"].split(",")
    return list(FLEET_ARCHITECTURES)


def build_flags() -> list[str]:
    # Float32 search arithmetic uses fast division/square root and FMA; float64 stays IEEE.
    flags = ["-O3", "-std=c++17", "--expt-relaxed-constexpr"]
    flags += [f"-gencode=arch=compute_{a[3:]},code={a}" for a in architectures()]
    newest = max(architectures(), key=lambda a: int(a[3:]))
    flags += [f"-gencode=arch=compute_{newest[3:]},code=compute_{newest[3:]}"]
    flags += ["-prec-div=false", "-prec-sqrt=false", "-ftz=true"]
    return flags + ["-shared", "-Xcompiler", "-fPIC"]


def library_path() -> Path:
    """Where this source's library is (or will be) built: named by source and flags."""
    flags = build_flags()
    digest = hashlib.sha256(SOURCE.read_bytes() + " ".join(flags).encode()).hexdigest()[:16]
    return build_dir() / f"s6_fit_kernel_{digest}.so"


def library():
    """Compile (once per source hash) and load the kernel library."""
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    flags = build_flags()
    target = library_path()
    if not target.is_file() and not Path(NVCC).is_file():
        raise RuntimeError(
            f"no GPU fit kernel {target.name} and no nvcc ({NVCC}) to build it: "
            "push it with `product_boxes setup`"
        )
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        # One compile per hash: concurrent component processes wait for the first one.
        with open(target.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not target.is_file():
                partial = target.with_suffix(f".{os.getpid()}.tmp")
                subprocess.run([NVCC, *flags, str(SOURCE), "-o", str(partial)], check=True)
                os.replace(partial, target)
    lib = ctypes.CDLL(str(target))
    for name, dtype in (("problem_size", PROBLEM), ("result_size", RESULT)):
        size = getattr(lib, name)()
        if size != dtype.itemsize:
            raise RuntimeError(f"{name} {size} != numpy {dtype.itemsize}")
    ptr = ctypes.c_void_p
    lib.s6_solve.restype = ctypes.c_int
    lib.s6_solve.argtypes = [ctypes.c_int, ptr, ctypes.c_int, ptr, ptr, ctypes.c_int, _Options]
    lib.s6_solve.argtypes += [ptr, ptr, ctypes.c_char_p]
    lib.s6_eval.restype = ctypes.c_int
    lib.s6_eval.argtypes = [ctypes.c_int, ctypes.c_int, ptr, ctypes.c_int, ptr, ptr, ctypes.c_int]
    lib.s6_eval.argtypes += [ctypes.c_int]
    lib.s6_eval.argtypes += [ptr, ptr, ptr, ctypes.c_char_p]
    lib.kernel_info.restype = ctypes.c_int
    lib.kernel_info.argtypes = [ctypes.c_int, ptr, ctypes.c_char_p]
    _LIBRARY = lib
    return lib


def kernel_info(device=None) -> dict:
    """Registers, shared/local bytes, resident blocks per SM and SM count of the solve kernel."""
    device = default_device() if device is None else device
    info = np.zeros(6, np.int32)
    message = _message()
    if library().kernel_info(device, info.ctypes.data, message):
        raise RuntimeError(message.value.decode())
    keys = ("registers_per_thread", "static_shared_bytes", "local_bytes_per_thread")
    keys += ("resident_blocks_per_sm", "sm_count", "threads_per_block")
    return dict(zip(keys, map(int, info)))


# --------------------------------------------------------------------------- CPU probe


class _Probe(Exception):
    pass


@dataclass
class CpuProblem:
    """The unchanged CPU refine's own closures at its SLSQP call."""

    cost: object
    objective_residual: object
    constraints: list
    x0: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    options: dict


def _closure(function, name):
    cells = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
    return cells[name].cell_contents


def probe(record: dict) -> CpuProblem:
    from cv.experiments.connected_shooting import real_exposure_replay as exposure

    captured = {}

    def fake_minimize(fun, x0, *args, **kwargs):
        captured.update(fun=fun, x0=np.array(x0, float), kwargs=kwargs)
        raise _Probe

    original = exposure.minimize
    exposure.minimize = fake_minimize
    try:
        with _record_context(record):
            exposure.refine(*record["args"], **record["kwargs"])
    except _Probe:
        pass
    finally:
        exposure.minimize = original
    if not captured:
        raise Unsupported("refine did not reach SLSQP (shared-state or early exit)")
    bounds = np.asarray(captured["kwargs"]["bounds"], float)
    return CpuProblem(
        cost=captured["fun"],
        objective_residual=_closure(captured["fun"], "objective_residual"),
        constraints=list(captured["kwargs"].get("constraints") or []),
        x0=captured["x0"],
        lo=bounds[:, 0],
        hi=bounds[:, 1],
        options=dict(captured["kwargs"].get("options") or {}),
    )


class _record_context:
    """Re-enter the switches recorded with the call: serve clearance, net-cord response mode and
    tape band, declared-net physical eligibility (absent keys keep the current values)."""

    def __init__(self, record):
        self.context = record.get("context") or {}

    def __enter__(self):
        from cv.experiments.connected_shooting import event_constraints, net_collision
        from cv.pipeline import net_cord_response as cord

        self.guard = event_constraints.serve_clearance(
            self.context.get("serve_clearance_scale_m", event_constraints.SERVE_CLEARANCE_SCALE_M)
        )
        self.guard.__enter__()
        self.tokens = []
        for var, key in (
            (cord._MODE, "net_cord_mode"),
            (cord._H_TOL, "net_cord_h_tol_m"),
            (net_collision._PHYSICAL_ELIGIBILITY, "net_physical_eligibility"),
        ):
            if key in self.context:
                self.tokens.append((var, var.set(self.context[key])))
        return self

    def __exit__(self, *exc):
        for var, token in reversed(self.tokens):
            var.reset(token)
        return self.guard.__exit__(*exc)


# --------------------------------------------------------------------------- packing


def _require(condition, reason):
    if not condition:
        raise Unsupported(reason)


def _scalar_pair(sigma):
    values = np.broadcast_to(np.asarray(sigma, float), (2,))
    return values.astype(float)


def _check_scene(scene, context: dict) -> None:
    """The dynamics, topology and surface features the kernel models (else ``Unsupported``)."""
    from cv.experiments.connected_shooting import ground_contact, passive_bounce
    from physics import bounce_reference, surface_model

    _require(context.get("passive_bounce", passive_bounce.active()) == "off", "passive bounce")
    _require(not context.get("ground_contact", ground_contact.active()), "ground settling mode")
    _require(scene.dynamics == "measured_240hz", "legacy dynamics")
    _require(scene.parameterization == "single_shooting", "shared contact states")
    _require(scene.rebound_mode == "point_scales", "fixed rebound")
    _require(scene.bounce_profile == "nominal", "bounce profile")
    _require(scene.bounce_regime_override is None, "regime override")
    _require(scene.terminal_net_tail is None, "terminal net tail")
    _require(getattr(scene, "observed_horizon_tail", None) is None, "observed horizon tail")
    _require(not scene.ground_settling, "ground settling")
    _require(scene.surface in bounce_reference.surfaces(), "surface spec")
    parsed = surface_model.parse(scene.surface)
    _require(parsed.inert and parsed.base_source == "reference", "region-aware surface")


def _fill_observations(problem, scene) -> tuple[int, np.ndarray]:
    """Contact frames, cadence and every native observation (camera, pixel, lens)."""
    n = len(scene.pixels)
    problem["contact_frames"][: n + 1] = np.asarray(scene.contact_frames, float)
    problem["fps"] = float(scene.fps)
    frames = [np.asarray(f, float) for f in scene.observation_frames]
    offsets = np.cumsum([0, *map(len, frames)])
    nobs = int(offsets[-1])
    _require(nobs <= MAX_OBS, "too many observations")
    problem["nobs"] = nobs
    problem["obs_off"][: n + 1] = offsets
    all_frames = np.concatenate(frames)
    problem["obs_frames"][:nobs] = all_frames
    problem["cameras"][: nobs * 12] = np.concatenate(
        [np.asarray(c, float) for c in scene.cameras]
    ).ravel()
    problem["pixels"][: nobs * 2] = np.concatenate(
        [np.asarray(p, float) for p in scene.pixels]
    ).ravel()
    if scene.camera_distortion is not None:
        problem["has_radial"] = 1
        problem["radial"][: nobs * 3] = np.concatenate(
            [np.asarray(r, float) for r in scene.camera_distortion]
        ).ravel()
    return nobs, all_frames


def _fill_nets(problem, scene, context: dict) -> int:
    """Supplied net-hit flights (net_collision) and the active net-cord response mode."""
    from cv.experiments.connected_shooting import net_collision
    from cv.pipeline import net_cord_response as cord
    from cv.pipeline import physics_knot_solver as knot

    mode = context.get("net_cord_mode", cord.active_mode())
    _require(mode in (cord.TAPE_CLIP, cord.ADMISSIBLE_SET), f"net cord mode {mode}")
    problem["net_mode"] = int(mode == cord.ADMISSIBLE_SET)
    problem["net_physical"] = int(
        context.get("net_physical_eligibility", net_collision._PHYSICAL_ELIGIBILITY.get())
    )
    problem["netk"] = [
        knot.NET_X_CENTER,
        knot.NET_HALF_WIDTH,
        knot.NET_H_CENTER,
        knot.NET_H_POST,
        knot.E_NET,
        knot.NET_TANG,
        context.get("net_cord_h_tol_m", cord.active_h_tol()),
        cord.POST_CONTACT_WIDENING,
    ]
    count = 0
    if scene.net_hit_frames is not None:
        for i, group in enumerate(scene.net_hit_frames):
            group = np.asarray(group, float)
            if len(group):
                problem["has_net"][i], problem["net_frame"][i] = 1, float(group[0])
                count += 1
    problem["any_net"] = int(count > 0)
    return count


def _fill_constants(problem, surface: str) -> None:
    """Measured bounce law, Cross spin, flight and net constants of this checkout."""
    from cv.experiments.connected_shooting import fast_flight, model
    from cv.pipeline import camera_cal
    from physics import bounce_reference, impact

    coefficients = bounce_reference.MEASURED[surface]
    problem["surf"][:8] = [
        coefficients["restitution_intercept_at_model_spin"],
        coefficients["restitution_slope_at_model_spin"],
        coefficients["retention_intercept_at_model_spin"],
        coefficients["retention_slope_at_model_spin"],
        impact.SURFACES[surface][0] / impact.HARD_COR,
        impact.SURFACES[surface][1],
        impact.E_X_COURT,
        impact.ALPHA,
    ]
    problem["phys"][:7] = [
        fast_flight.G,
        fast_flight.K,
        fast_flight.C_DRAG,
        fast_flight.LIFT_SCALE,
        fast_flight.SPIN_DECAY,
        model.R_BALL,
        bounce_reference.DWELL_SECONDS,
    ]
    problem["net"][:4] = [
        camera_cal.NET_H_CENTER,
        camera_cal.NET_H_POST,
        camera_cal.net_support_half_span(),
        camera_cal.COURT_W,
    ]


def pack(record: dict, cpu: CpuProblem | None = None) -> np.ndarray:
    """One ``Problem`` for a captured refine call (raises ``Unsupported``)."""
    names = ("scene", "initial", "bounces", "native", "axes", "duration", "maxiter")
    call = dict(zip(names, record["args"])) | dict(record["kwargs"])
    return pack_call(call, record.get("context") or {}, lambda: cpu or probe(record))


def pack_call(call: dict, context: dict, cpu_problem) -> np.ndarray:
    """One ``Problem`` for refine's named arguments; ``cpu_problem()`` supplies its closures."""
    from cv.experiments.connected_shooting import (
        event_constraints,
        model,
    )

    scene = call["scene"]
    n = len(scene.pixels)
    _check_scene(scene, context)
    _require(call.get("duration") is None, "exposure duration model")
    for name in (
        "first_contact_epoch_prior",
        "interior_contact_epoch_plan",
        "observation_net_seed",
        "net_seed_candidate",
    ):
        _require(call.get(name) is None, name)
    for name in ("shared_contact_states", "local_flight_refits", "_net_mesh_response"):
        _require(not call.get(name, False), name)
    _require(n <= MAX_F - 1 and 3 + 6 * n + 2 <= MAX_P, "too many flights")
    surface = scene.surface
    cpu = cpu_problem()

    problem = np.zeros((), PROBLEM)
    initial = np.asarray(call["initial"], float)
    P = 3 + 6 * n + 2
    _require(len(cpu.x0) == P and initial.shape == (P,), "timing variables")
    scaling = call.get("parameter_scaling", "global")
    if scaling == "global":
        scale = np.r_[[10] * 3, [30] * (3 * n), [3] * (3 * n), [1, 1]].astype(float)
    elif scaling == "accepted_candidate_local":
        scale = np.maximum(
            np.abs(initial), np.r_[[1] * 3, [10] * (3 * n), [1] * (3 * n), [0.1, 0.1]]
        )
    else:
        raise Unsupported("parameter scaling")
    problem["P"], problem["n"] = P, n
    problem["scale"][:P] = scale
    problem["lo"][:P] = cpu.lo
    problem["hi"][:P] = cpu.hi
    nobs, all_frames = _fill_observations(problem, scene)
    net_flights = _fill_nets(problem, scene, context)
    bounces = event_constraints.validate(
        scene, call["bounces"], call.get("bounce_bracket_frames", 1.0)
    )
    for i, group in enumerate(bounces):
        _require(len(group) <= MAX_EXP, "too many expected bounces")
        problem["n_exp"][i] = len(group)
        problem["exp_frames"][i * MAX_EXP : i * MAX_EXP + len(group)] = group
    problem["uncertainty"] = float(call.get("bounce_bracket_frames", 1.0))
    serve = context.get("serve_clearance_scale_m", event_constraints.SERVE_CLEARANCE_SCALE_M)
    problem["net_scale"] = event_constraints.NET_CLEARANCE_GUIDANCE_M
    problem["net_scale0"] = (
        serve if serve is not None else event_constraints.NET_CLEARANCE_GUIDANCE_M
    )
    problem["last_frame"] = float(call["native"][-1][-1])

    # constraints (penalty rows)
    inequality = call.get("inequality_constraints", True)
    directional = call.get("directional_inequalities", True)
    limit = call.get("directional_rms_limit_px")
    margin = call.get("directional_interior_margin_px", 0.5)
    windows = []
    if limit is not None:
        for event in np.asarray(call.get("directional_event_frames") or [], float):
            for horizon in call.get("directional_horizons", (4, 10)):
                for low, high in ((event - horizon, event), (event, event + horizon)):
                    mask = (low <= all_frames) & (all_frames <= high)
                    if np.any(mask):
                        index = np.flatnonzero(mask)
                        _require(index[-1] - index[0] + 1 == len(index), "noncontiguous window")
                        windows.append((index[0], index[-1] + 1))
    _require(len(windows) <= MAX_WIN, "too many directional windows")
    use_dir = bool(inequality and directional and windows)
    problem["use_dir_pen"] = int(use_dir)
    problem["nwin"] = len(windows) if use_dir else 0
    for w, (a, b) in enumerate(windows if use_dir else []):
        problem["win_lo"][w], problem["win_hi"][w] = a, b
    if limit is not None:
        problem["dir_limit2"] = (float(limit) - float(margin)) ** 2
    problem["term_margin0"] = 0.01 if windows else 0.0
    kind = (
        None if not isinstance(scene.supported_ending, dict) else scene.supported_ending.get("kind")
    )
    if (
        model.original_contact_boundary(scene)
        or kind in PARTIAL_ENDINGS
        or (
            kind in {"fov_exit", "last_visible_sample"}
            and scene.supported_ending.get("partial_flight") is True
        )
    ):
        problem["term_kind"] = 0
    else:
        if kind == "net_stop":
            problem["term_kind"] = 3
            target = scene.supported_ending.get("net_frame")
            if target is not None:
                problem["has_stop_target"], problem["stop_target"] = 1, float(target)
        else:
            _require(len(bounces[-1]) > 0 or not inequality, "terminal slack without bounce")
            problem["term_kind"] = 1
    rebound = call.get("terminal_rebound_frames")
    if rebound is not None:
        # event_constraints terminal_rebound_penalty row and terminal_rebound_slack
        frames = event_constraints.validate_terminal_rebound(scene, bounces, rebound)
        problem["has_rebound"], problem["rebound_first"] = 1, float(frames[0])
        problem["term_kind"] = 2
    problem["use_term_pen"] = int(bool(inequality))
    expected_constraints = int(bool(inequality)) + int(use_dir)
    _require(len(cpu.constraints) == expected_constraints, "unexpected constraint set")

    # priors and anchors
    target = call.get("first_contact_target")
    if target is not None:
        problem["has_ctarget"] = 1
        problem["ctarget"] = np.asarray(target[0], float)
        problem["ctarget_sigma"] = np.asarray(target[1], float)
    anchors = call.get("anchor_targets")
    ops = []
    if anchors is not None:
        for i, (group, contact) in enumerate(
            zip(anchors["bounce"], anchors["contact"], strict=True)
        ):
            for j, row in enumerate(group):
                if row is None:
                    continue
                ops.append(
                    (ANCHOR_BOUNCE, i, j, np.asarray(row[0], float)[:2], _scalar_pair(row[1]))
                )
            if contact is not None:
                ops.append(
                    (
                        ANCHOR_CONTACT,
                        i,
                        0,
                        np.asarray(contact[0], float)[:2],
                        _scalar_pair(contact[1]),
                    )
                )
        closing = anchors.get("closing_contact")
        if closing is not None:
            ops.append(
                (
                    ANCHOR_CLOSING,
                    n - 1,
                    0,
                    np.asarray(closing["court_xy_m"], float),
                    _scalar_pair(closing["sigma_m"]),
                )
            )
    _require(len(ops) <= MAX_ANCHOR, "too many anchors")
    problem["n_anchor"] = len(ops)
    for a, (kind_, flight, ordinal, xy, sigma) in enumerate(ops):
        problem["anchor_kind"][a], problem["anchor_flight"][a] = kind_, flight
        problem["anchor_ordinal"][a] = ordinal
        problem["anchor_xy"][2 * a : 2 * a + 2] = xy
        problem["anchor_sigma"][2 * a : 2 * a + 2] = sigma
    priors = call.get("bounce_epoch_priors") or []
    _require(len(priors) <= MAX_EP, "too many epoch priors")
    problem["n_ep"] = len(priors)
    for e, row in enumerate(priors):
        problem["ep_flight"][e], problem["ep_ordinal"][e] = (
            int(row["flight_index"]),
            int(row["ordinal"]),
        )
        problem["ep_frame"][e], problem["ep_sigma"][e] = (
            float(row["frame"]),
            float(row["sigma_frames"]),
        )
    cue = call.get("first_contact_depth_cue")
    cue_rows = 0
    if cue is not None and cue["cue"]["status"] == "supported":
        row = cue["cue"]
        problem["has_cue"], cue_rows = 1, 1
        problem["cue_root"] = np.asarray(row["root_xy"], float)
        problem["cue_axis"] = np.asarray(row["axis"], float)
        problem["cue_sigma"], problem["cue_weight"] = (
            float(row["sigma_m"]),
            float(row["pixel_weight"]),
        )

    _fill_constants(problem, surface)
    objective_rows = 2 * nobs + sum(len(g) + 2 for g in bounces) + 2 + int(problem["has_rebound"])
    objective_rows += 3 * net_flights  # four collision rows replace one clearance row
    objective_rows += 3 * int(target is not None) + 2 * len(ops) + len(priors) + cue_rows
    problem["M_objective"] = objective_rows
    problem["soft_rows"] = 2 * nobs
    problem["image_scale"] = 1.0
    problem["M"] = (
        objective_rows + 3 * int(problem["use_term_pen"]) + (len(windows) if use_dir else 0)
    )
    return problem


SEED_SCALE = np.array([30.0, 30.0, 30.0, 3.0, 3.0, 3.0])


def pack_seed(residual, lower, upper) -> np.ndarray:
    """One kind-1 ``Problem`` for ``initialization.anchor_connected_seed``'s local residual.

    Reads the flight's own bound arguments (start, next-contact anchor, one-flight subscene,
    expected bounces, ground anchors) and the walk's rebound corrections from the closure;
    raises ``Unsupported`` for anything the kernel does not model.
    """
    code = residual.__code__
    names = code.co_varnames[: code.co_argcount]
    defaults = dict(zip(names[len(names) - len(residual.__defaults__) :], residual.__defaults__))
    cells = dict(zip(code.co_freevars, (c.cell_contents for c in residual.__closure__ or ())))
    _require(
        {"start_xyz", "target_xyz", "subscene", "expected_bounces", "anchors"} <= set(defaults),
        "seed residual signature",
    )
    _require(cells.get("exposure_duration") is None, "exposure duration model")
    scene = defaults["subscene"]
    _require(len(scene.pixels) == 1, "seed walk of one flight")
    _check_scene(scene, {})
    parameters = np.asarray(cells["parameters"], float)
    expected = np.asarray(defaults["expected_bounces"], float)
    _require(len(expected) <= MAX_EXP, "too many expected bounces")
    problem = np.zeros((), PROBLEM)
    problem["kind"], problem["n"], problem["P"] = 1, 1, 6
    problem["scale"][:6] = SEED_SCALE
    problem["lo"][:6] = np.asarray(lower, float) / SEED_SCALE
    problem["hi"][:6] = np.asarray(upper, float) / SEED_SCALE
    problem["fixed_start"] = np.asarray(defaults["start_xyz"], float)
    problem["fixed_scales"] = parameters[-2:]
    target = defaults["target_xyz"]
    if target is not None:
        problem["has_init_target"] = 1
        problem["init_target"] = np.asarray(target, float)
    problem["init_target_sigma"] = 0.30
    problem["n_exp"][0] = len(expected)
    problem["exp_frames"][: len(expected)] = expected
    ops = [
        (ordinal, np.asarray(anchor[0], float)[:2], _scalar_pair(anchor[1]))
        for ordinal, anchor in enumerate(defaults["anchors"])
        if anchor is not None
    ]
    _require(len(ops) <= MAX_ANCHOR, "too many anchors")
    problem["n_anchor"] = len(ops)
    for a, (ordinal, xy, sigma) in enumerate(ops):
        problem["anchor_kind"][a], problem["anchor_ordinal"][a] = ANCHOR_BOUNCE, ordinal
        problem["anchor_xy"][2 * a : 2 * a + 2] = xy
        problem["anchor_sigma"][2 * a : 2 * a + 2] = sigma
    nobs, _ = _fill_observations(problem, scene)
    _fill_nets(problem, scene, {})  # a net-hit flight runs net_collision, as model.chain does
    _fill_constants(problem, scene.surface)
    problem["net_scale0"] = problem["net_scale"] = 1.0
    problem["image_scale"] = 8.0
    rows = 2 * nobs + 3 * int(target is not None) + len(expected) + 1 + 2 * len(ops)
    problem["M"] = problem["M_objective"] = problem["soft_rows"] = rows
    return problem


WALK_BLUR_PX = 1.5  # real_exposure_replay.prediction -> swept.predict(..., 1.5)


def pack_walk(residual, lower, upper) -> np.ndarray:
    """One kind-1 swept ``Problem`` for ``structure_aware_connected_seed``'s per-flight residual.

    Rows: swept-exposure image tips / 8, next-hitter endpoint / 0.10 m, supplied-bounce timing
    / 0.25 frame (20 when missing) and the extra-bounce row, all soft-L1 (f_scale 2). Reads the
    closure of the walk's ``residual``; raises ``Unsupported`` for anything not modelled.
    """
    from cv.experiments.connected_shooting import model

    cells = dict(
        zip(
            residual.__code__.co_freevars,
            (c.cell_contents for c in residual.__closure__ or ()),
            strict=True,
        )
    )
    needed = {"subscene", "start_xyz", "parameters", "axis_groups", "index", "target_xyz"}
    _require(needed | {"expected_bounces", "exposure_duration"} <= set(cells), "walk residual")
    duration = cells["exposure_duration"]
    swept = duration is not None  # None: point projections (leading.image_prediction)
    if swept:
        duration = float(duration)
        _require(0 < duration <= 1, "exposure duration outside (0, 1] frame")
    scene = cells["subscene"]
    _require(len(scene.pixels) == 1, "walk of one flight")
    _check_scene(scene, {})
    options = cells.get("prediction_options") or {}
    termination = options.get("termination_kind", "terminal_bounce")
    if not swept:
        extra = 0.0
    elif termination in ("terminal_net_impact", "net_stop", "observed_horizon", "original_contact"):
        extra = 0.0
    else:
        _require(termination in ("terminal_bounce", "second_bounce"), "imaging termination")
        _require(not model.original_contact_boundary(scene), "original-contact imaging")
        extra = 1.0
    frames = np.asarray(scene.observation_frames[0], float)
    first, last = float(scene.contact_frames[0]), float(scene.contact_frames[-1]) + extra
    if swept:
        # swept_exposure.windows: an uncovered exposure makes every residual the 1e5 sentinel
        _require(
            bool(np.all(frames >= first) and np.all(frames + duration <= last)),
            "exposure window outside the imaging domain",
        )
        axes = np.asarray(cells["axis_groups"][cells["index"]], float)
        _require(axes.shape == (len(frames), 2), "one image axis per observation")
        norms = np.linalg.norm(axes, axis=1)
        _require(bool(np.all(norms >= 1e-9)), "zero image axis")
    parameters = np.asarray(cells["parameters"], float)
    expected = np.asarray(cells["expected_bounces"], float)
    _require(len(expected) <= MAX_EXP, "too many expected bounces")
    problem = np.zeros((), PROBLEM)
    problem["kind"], problem["n"], problem["P"], problem["swept"] = 1, 1, 6, int(swept)
    problem["scale"][:6] = SEED_SCALE
    problem["lo"][:6] = np.asarray(lower, float) / SEED_SCALE
    problem["hi"][:6] = np.asarray(upper, float) / SEED_SCALE
    problem["fixed_start"] = np.asarray(cells["start_xyz"], float)
    problem["fixed_scales"] = parameters[-2:]
    target = cells["target_xyz"]
    if target is not None:
        problem["has_init_target"] = 1
        problem["init_target"] = np.asarray(target, float)
    problem["init_target_sigma"] = 0.10
    problem["n_exp"][0] = len(expected)
    problem["exp_frames"][: len(expected)] = expected
    nobs, _ = _fill_observations(problem, scene)
    _require(not (_fill_nets(problem, scene, {}) and swept), "swept exposure on a net-hit flight")
    if swept:
        problem["axes"][: 2 * nobs] = (axes / norms[:, None]).ravel()
        problem["swept_duration"], problem["swept_blur"] = duration, WALK_BLUR_PX
        problem["imaging_extra"] = extra
    _fill_constants(problem, scene.surface)
    problem["net_scale0"] = problem["net_scale"] = 1.0
    problem["image_scale"] = 8.0
    rows = 2 * nobs + 3 * int(target is not None) + len(expected) + 1
    problem["M"] = problem["M_objective"] = problem["soft_rows"] = rows
    return problem


# --------------------------------------------------------------------------- launches


def multistart(x0, lo, hi, count, rng):
    """``count`` starts: the recorded seed, then bounded perturbations of it (scaled space).

    Half of the extra starts move each free coordinate by 1% of its bound width, half by 5%;
    fixed coordinates (lo == hi) never move. These are search starts only; every candidate is
    judged by the same objective and constraints.
    """
    x0 = np.clip(np.asarray(x0, float), lo, hi)
    width = np.where(hi > lo, hi - lo, 0.0)
    starts = [x0]
    for k in range(1, count):
        sigma = 0.01 if k % 2 else 0.05
        starts.append(np.clip(x0 + rng.normal(size=len(x0)) * sigma * width, lo, hi))
    return starts


def _message():
    return ctypes.create_string_buffer(256)


def solve(problems, index, seeds, *, device=None, **options):
    """Solve seed ``k`` of problem ``index[k]`` (see ``solve_local``).

    With ``TENNIS_S6_GPU_FIT_SERVER=SOCKET`` the launch goes to the shared fit server
    (``server.py``), which merges concurrent clients' requests into one kernel launch;
    ``off`` (the queue's rollback) or unset: this process's own CUDA context.
    """
    device = default_device() if device is None else device
    socket_path = server_socket()
    if socket_path:
        from cv.experiments.gpu_fit import server

        answer = server.request(socket_path, problems, index, seeds, options)
    else:
        answer = solve_local(problems, index, seeds, device=device, **options)
    # Budget clock (TENNIS_S6_WORK_BUDGET): the launch's objective evaluations, deterministic.
    work_clock.advance(
        work_clock.GPU_EVALUATION_UNITS * float(np.sum(answer[0]["evaluations"])), "gpu_solve"
    )
    return answer


def solve_local(
    problems,
    index,
    seeds,
    *,
    device=None,
    max_iter=200,
    max_iter64=0,
    ftol=1e-7,
    ftol64=1e-10,
    lambda0=1e-3,
    small=3,
    central32=True,
    report64=False,
):
    """Solve seed ``k`` of problem ``index[k]``; returns (results, kernel_ms, total_ms).

    ``max_iter`` float32 search iterations, then ``max_iter64`` float64 finishing iterations.
    """
    device = default_device() if device is None else device
    lib = library()
    problems = np.ascontiguousarray(problems, PROBLEM)
    index = np.ascontiguousarray(index, np.int32)
    q = np.zeros((len(index), MAX_P))
    for k, seed in enumerate(seeds):
        q[k, : len(seed)] = seed
    results = np.zeros(len(index), RESULT)
    options = _Options(
        max_iter=max_iter,
        max_iter64=max_iter64,
        m_stride=int(problems["M"].max()) + 8,
        p_stride=int(problems["P"].max()) ** 2 + int(problems["P"].max()),
        ftol=ftol,
        ftol64=ftol64,
        lambda0=lambda0,
        small_steps_to_stop=small,
        central32=int(central32),
        report64=int(report64),
    )
    timing = np.zeros(2)
    message = _message()
    rc = lib.s6_solve(
        device,
        problems.ctypes.data,
        len(problems),
        index.ctypes.data,
        q.ctypes.data,
        len(index),
        options,
        results.ctypes.data,
        timing.ctypes.data,
        message,
    )
    if rc:
        raise RuntimeError(message.value.decode())
    return results, float(timing[0]), float(timing[1])


def raw_constraint_count(problem) -> int:
    """Raw SLSQP constraint values the kernel reports: terminal (3) then directional windows."""
    return 3 * int(problem["use_term_pen"]) + int(problem["nwin"]) * int(problem["use_dir_pen"])


def server_socket() -> str | None:
    """The shared fit server's socket (``TENNIS_S6_GPU_FIT_SERVER``), or None (``off``/unset)."""
    socket_path = os.environ.get("TENNIS_S6_GPU_FIT_SERVER", "off")
    return None if socket_path in ("", "off") else socket_path


def evaluate(problem, qs, *, device=None, precision=64, raw=False):
    """Residual rows, (objective cost, penalized cost) and success at scaled points.

    ``raw=True`` also returns the raw constraint values (``raw_constraint_count`` columns).
    With the fit server set, the evaluation runs in its context: no client opens one."""
    socket_path = server_socket()
    if socket_path:
        from cv.experiments.gpu_fit import server

        answer = server.evaluate_request(socket_path, problem, qs, precision, raw)
    else:
        answer = evaluate_local(problem, qs, device=device, precision=precision, raw=raw)
    work_clock.advance(work_clock.GPU_EVALUATION_UNITS * len(np.atleast_2d(qs)), "gpu_evaluate")
    return answer


def evaluate_local(problem, qs, *, device=None, precision=64, raw=False):
    """``evaluate`` in this process's own CUDA context."""
    device = default_device() if device is None else device
    lib = library()
    problems = np.ascontiguousarray(np.atleast_1d(problem), PROBLEM)
    qs = np.atleast_2d(np.asarray(qs, float))
    padded = np.zeros((len(qs), MAX_P))
    padded[:, : qs.shape[1]] = qs
    index = np.zeros(len(qs), np.int32)
    nraw = raw_constraint_count(problems[0]) if raw else 0
    stride = int(problems["M"].max()) + (nraw + 8 if raw else 8)
    rows = np.zeros((len(qs), stride))
    costs = np.zeros((len(qs), 2))
    ok = np.zeros(len(qs), np.int32)
    message = _message()
    rc = lib.s6_eval(
        device,
        precision,
        problems.ctypes.data,
        len(problems),
        index.ctypes.data,
        padded.ctypes.data,
        len(qs),
        stride,
        rows.ctypes.data,
        costs.ctypes.data,
        ok.ctypes.data,
        message,
    )
    if rc:
        raise RuntimeError(message.value.decode())
    M = int(problems["M"][0])
    if raw:
        return rows[:, :M], costs, ok.astype(bool), rows[:, M : M + nraw]
    return rows[:, :M], costs, ok.astype(bool)


def evaluate_many(requests, *, device=None, precision=64):
    """Several ``evaluate_local`` calls of one precision in one launch.

    ``requests`` is a list of ``(problem, qs, raw)``; the answer for each is exactly what
    ``evaluate_local(problem, qs, precision=precision, raw=raw)`` returns: the kernel evaluates
    every point in its own thread from its own problem, and the row stride only leaves room.
    """
    device = default_device() if device is None else device
    lib = library()
    problems = np.ascontiguousarray(
        np.concatenate([np.atleast_1d(problem)[:1] for problem, _, _ in requests]), PROBLEM
    )
    blocks = [np.atleast_2d(np.asarray(qs, float)) for _, qs, _ in requests]
    count = sum(len(block) for block in blocks)
    padded = np.zeros((count, MAX_P))
    index = np.zeros(count, np.int32)
    cursor = 0
    for k, block in enumerate(blocks):
        padded[cursor : cursor + len(block), : block.shape[1]] = block
        index[cursor : cursor + len(block)] = k
        cursor += len(block)
    widths = [int(p["M"]) + raw_constraint_count(p) for p in problems]
    stride = max(widths) + 8
    rows = np.zeros((count, stride))
    costs = np.zeros((count, 2))
    ok = np.zeros(count, np.int32)
    message = _message()
    rc = lib.s6_eval(
        device,
        precision,
        problems.ctypes.data,
        len(problems),
        index.ctypes.data,
        padded.ctypes.data,
        count,
        stride,
        rows.ctypes.data,
        costs.ctypes.data,
        ok.ctypes.data,
        message,
    )
    if rc:
        raise RuntimeError(message.value.decode())
    answers, cursor = [], 0
    for problem, block, (_, _, raw) in zip(problems, blocks, requests, strict=True):
        part = slice(cursor, cursor + len(block))
        cursor += len(block)
        M, nraw = int(problem["M"]), raw_constraint_count(problem)
        answer = (rows[part, :M].copy(), costs[part].copy(), ok[part].astype(bool))
        answers.append(answer + (rows[part, M : M + nraw].copy(),) if raw else answer)
    return answers


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Build (once) and load the GPU fit kernel library."
    )
    parser.add_argument(
        "--check", type=int, metavar="DEVICE", help="also query the kernel on DEVICE"
    )
    args = parser.parse_args()
    print(library()._name)
    if args.check is not None:  # loads on the card: a driver too old for the build fails here
        print(json.dumps(kernel_info(args.check)))
