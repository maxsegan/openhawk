"""Pre-imported fork server for the S6 stage subprocesses (``TENNIS_S6_STAGE_FORKSERVER``).

Every S6 attempt and component runs ``python -m cv.pipeline.s6_labeled_stage`` in its own
process (``s6_attempt_execution.execute``, ``s6_broadcast_backend.run_stage``): mutable
numerical state never crosses attempt boundaries. A match starts ~2,700 of them, and each
spends ~0.5 CPU-s starting Python and importing the same ~1,000 modules before its fit.

``on``: the calling process keeps one server per (environment, working directory). The
server is a child process started with exactly the stage's environment and working
directory; it imports the stage's modules once and forks one fresh child per stage run. The
child resets its signal handlers, starts its own session, sets its argv, stdin
(``/dev/null``), stdout/stderr (the stage log) and runs the module as ``__main__`` like
``-m``, so each run still gets a new process with untouched module state (the server never
runs a stage, opens a GPU context or connects to the fit server). The server reaps each
child (``wait4``), so its CPU reaches the caller's ``RUSAGE_CHILDREN`` once the server
exits; servers stop at the caller's exit and are waited for.

``off`` (default): ``subprocess`` starts a new interpreter per stage run (rollback).
A server that cannot start or dies falls back to a new interpreter for that run.
"""

from __future__ import annotations

import atexit
import json
import os
import select
import signal
import subprocess
import sys
import threading
from pathlib import Path

SWITCH = "TENNIS_S6_STAGE_FORKSERVER"
MODULE = "cv.pipeline.s6_labeled_stage"
#: The stage records every loaded module under these prefixes as its implementation
#: (``s6_labeled_stage.implementation_records``), so the server must not load any of them:
#: the child then loads exactly the modules a fresh interpreter would, and its receipts are
#: byte-identical. The server refuses to serve if a preload pulled one in.
IDENTITY_PREFIXES = ("cv.experiments.connected_shooting.", "physics.", "cv.pipeline.s6_")
#: What a stage run imports outside those prefixes (third-party numerics and the shared
#: pipeline modules, ~0.40 of ~0.48 CPU-s of a stage's imports); none of them imports an
#: identity module (checked module by module on a stage's full import list).
PRELOAD = (
    "numpy",
    "scipy.constants",
    "scipy.fft",
    "scipy.linalg",
    "scipy.optimize",
    "scipy.sparse",
    "scipy.spatial",
    "scipy.special",
    "cv2",
    "PIL.Image",
    "PIL.ImageDraw",
    "PIL.ImageFont",
    "cv.experiments.gpu_fit.cooperative",
    "cv.experiments.gpu_fit.refine_hook",
    "cv.experiments.gpu_fit.seed_hook",
    "cv.pipeline.artifact_cache",
    "cv.pipeline.bounce_detect",
    "cv.pipeline.camera_P_per_frame",
    "cv.pipeline.camera_artifacts",
    "cv.pipeline.camera_bundle",
    "cv.pipeline.camera_cal",
    "cv.pipeline.camera_project",
    "cv.pipeline.camera_scope",
    "cv.pipeline.contact_frame_refiner",
    "cv.pipeline.contacts_audio",
    "cv.pipeline.court",
    "cv.pipeline.court_geometry_gate",
    "cv.pipeline.court_gpu",
    "cv.pipeline.court_near_baseline_refinement",
    "cv.pipeline.court_paint_refinement",
    "cv.pipeline.court_topology",
    "cv.pipeline.event_contact_witness",
    "cv.pipeline.event_topology",
    "cv.pipeline.flight_anchors",
    "cv.pipeline.flight_ledger",
    "cv.pipeline.frame_identity",
    "cv.pipeline.paths",
    "cv.pipeline.physics_interpretation",
    "cv.pipeline.physics_knot_solver",
    "cv.pipeline.pipeline_evidence",
    "cv.pipeline.player_court_v2",
    "cv.pipeline.player_identity",
    "cv.pipeline.player_side_association",
    "cv.pipeline.player_tracker",
    "cv.pipeline.point_context",
    "cv.pipeline.point_grammar",
    "cv.pipeline.pose",
    "cv.pipeline.provenance",
    "cv.pipeline.resolution",
    "cv.pipeline.rich_ball_physics",
    "cv.pipeline.run_manifest",
    "cv.pipeline.score_grammar",
    "cv.pipeline.score_vlm",
    "cv.pipeline.serve_location_prior",
    "cv.pipeline.serve_speed_graphic",
    "cv.pipeline.switch_fallbacks",
    "cv.pipeline.track_heal",
    "cv.pipeline.trajectory_contract",
    "cv.pipeline.vlm_policy",
    "cv.validation.current_standard_event_truth",
    "cv.validation.hawkeye_flight_fit",
    "cv.validation.playstyle_pattern",
    "cv.validation.s6_owner_camera_transport",
    "cv.validation.s6_owner_court_audit",
    "cv.validation.s6_owner_ground_camera",
    "cv.validation.s6_owner_inputs",
    "cv.validation.score_cross_match_event_labels_v5",
    "cv.validation.wk3_s6_cohort",
    "scripts.shared_data",
)


def identity_modules() -> list[str]:
    """Loaded modules the stage would record as its implementation."""
    return sorted(name for name in sys.modules if name.startswith(IDENTITY_PREFIXES))


def enabled() -> bool:
    value = os.environ.get(SWITCH, "off")
    if value not in ("on", "off"):
        raise ValueError(f"{SWITCH} must be on or off")
    return value == "on"


# --- server side ---------------------------------------------------------------------------


def _child(request: dict, reply_fd: int) -> None:
    """In the forked child: become ``python -m MODULE ARGS`` and never return."""
    code = 1
    try:
        os.close(reply_fd)
        for number in (signal.SIGCHLD, signal.SIGTERM, signal.SIGINT, signal.SIGPIPE):
            signal.signal(number, signal.SIG_DFL)
        os.setsid()
        null = os.open(os.devnull, os.O_RDONLY)
        log = os.open(request["log"], os.O_WRONLY | os.O_APPEND)
        os.dup2(null, 0)
        os.dup2(log, 1)
        os.dup2(log, 2)
        os.close(null)
        os.close(log)
        sys.stdin = open(0, closefd=False)
        sys.stdout = open(1, "w", closefd=False)
        sys.stderr = open(2, "w", closefd=False)
        import runpy

        sys.argv = ["-m", *request["args"]]
        try:
            runpy.run_module(request["module"], run_name="__main__", alter_sys=True)
            code = 0
        except SystemExit as stop:
            code = stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
            if not isinstance(stop.code, (int, type(None))):
                print(stop.code, file=sys.stderr)
        except BaseException:  # noqa: BLE001 - reported like an uncaught exception
            import traceback

            traceback.print_exc()
            code = 1
    finally:
        # A normal interpreter exit: non-daemon threads joined, atexit handlers run, streams
        # flushed; os._exit then skips the server's own frames.
        try:
            threading._shutdown()
            import atexit as exit_handlers

            exit_handlers._run_exitfuncs()
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(code)


def serve() -> None:
    """Read requests (one JSON line each) on stdin; answer ``started`` and ``exited`` lines."""
    import importlib

    for name in PRELOAD:
        try:
            importlib.import_module(name)
        except ImportError:
            pass
    if loaded := identity_modules():
        sys.stderr.write(f"preload loaded implementation modules {loaded[:5]}: not serving\n")
        sys.exit(3)
    from cv.pipeline import paths, provenance

    # Every stage records the checkout's git identity: once here, inherited by each child.
    provenance.git_record(paths.REPO_ROOT)
    reply = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)  # a stray print must not corrupt replies
    reply.write(json.dumps({"ready": os.getpid()}) + "\n")
    children: dict[int, str] = {}
    buffer = b""
    open_input = True
    while open_input or children:
        readable, _, _ = select.select([0] if open_input else [], [], [], 0.02)
        if readable:
            chunk = os.read(0, 65536)
            if not chunk:
                open_input = False
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                request = json.loads(line)
                pid = os.fork()
                if pid == 0:
                    _child(request, reply.fileno())
                children[pid] = request["id"]
                reply.write(json.dumps({"id": request["id"], "started": pid}) + "\n")
        while children:
            pid, status, _usage = os.wait4(-1, os.WNOHANG)
            if pid == 0:
                break
            ident = children.pop(pid, None)
            if ident is not None:
                code = os.waitstatus_to_exitcode(status)
                reply.write(json.dumps({"id": ident, "exited": code}) + "\n")


# --- client side ---------------------------------------------------------------------------


class _Server:
    def __init__(self, env: dict, cwd: str) -> None:
        self.process = subprocess.Popen(
            [sys.executable, "-m", __name__],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
            cwd=cwd,
        )
        self.condition = threading.Condition()
        self.replies: dict[str, dict] = {}
        self.dead = False
        ready = self.process.stdout.readline()
        if not ready or "ready" not in json.loads(ready):
            self.process.wait()
            raise OSError("S6 stage fork server did not start")
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.process.stdout:
            message = json.loads(line)
            with self.condition:
                self.replies.setdefault(message["id"], {}).update(message)
                self.condition.notify_all()
        with self.condition:
            self.dead = True
            self.condition.notify_all()

    def submit(self, ident: str, request: dict) -> None:
        with self.condition:
            if self.dead:
                raise OSError("S6 stage fork server died")
            self.process.stdin.write((json.dumps({"id": ident, **request}) + "\n").encode())
            self.process.stdin.flush()

    def await_key(self, ident: str, key: str, timeout: float | None = None):
        """The reply's ``key`` once it arrived; TimeoutExpired / OSError otherwise."""
        with self.condition:
            arrived = self.condition.wait_for(
                lambda: key in self.replies.get(ident, {}) or self.dead, timeout
            )
            reply = self.replies.get(ident, {})
            if key in reply:
                if key == "exited":
                    self.replies.pop(ident, None)
                return reply[key]
            if not arrived:
                raise subprocess.TimeoutExpired(MODULE, timeout)
            raise OSError("S6 stage fork server died while a stage ran")

    def stop(self) -> None:
        try:
            self.process.stdin.close()
        except OSError:
            pass
        self.process.wait()


_SERVERS: dict[tuple, _Server] = {}
_SERVERS_LOCK = threading.Lock()
_COUNTER = iter(range(1, 1 << 62))


def _server(env: dict, cwd: str) -> _Server:
    key = (json.dumps(sorted(env.items())), cwd)
    with _SERVERS_LOCK:
        server = _SERVERS.get(key)
        if server is None or server.dead:
            if server is not None:
                server.stop()
            if not _SERVERS:
                atexit.register(shutdown)
            server = _SERVERS[key] = _Server(env, cwd)
        return server


def shutdown() -> None:
    """Stop every server of this process and wait for it (its children's CPU is then ours)."""
    with _SERVERS_LOCK:
        servers = list(_SERVERS.values())
        _SERVERS.clear()
    for server in servers:
        server.stop()


class StageProcess:
    """``subprocess.Popen``-like handle on one stage run (``pid``, ``wait``, ``returncode``)."""

    def __init__(
        self, args: list[str], *, log: Path, env: dict, cwd: str | None = None, module=MODULE
    ) -> None:
        self.returncode = None
        self._server = _server(dict(env), str(cwd or os.getcwd()))
        self._id = f"{os.getpid()}-{next(_COUNTER)}"
        self._server.submit(self._id, {"module": module, "args": args, "log": str(log)})
        self.pid = self._server.await_key(self._id, "started")

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            self.returncode = self._server.await_key(self._id, "exited", timeout)
        return self.returncode


def popen(command: list[str], *, stream, log: Path, env: dict, cwd=None, **popen_options):
    """A stage run: forked by the server when the switch is on and ``command`` is the stage
    module, else (or when the server fails) ``subprocess.Popen`` writing to ``stream``."""
    args = stage_command(command) if enabled() else None
    if args is not None:
        try:
            return StageProcess(args, log=log, env=env, cwd=cwd)
        except OSError as error:
            from cv.pipeline import switch_fallbacks

            switch_fallbacks.record(
                SWITCH, "server_failed", f"{type(error).__name__}: {error}", "s6 stage"
            )
    return subprocess.Popen(
        command, stdout=stream, stderr=subprocess.STDOUT, env=env, cwd=cwd, **popen_options
    )


def stage_command(command: list[str]) -> list[str] | None:
    """The module arguments of ``[python, -m, MODULE, ...]``, or None for any other command."""
    if len(command) >= 3 and command[0] == sys.executable and command[1:3] == ["-m", MODULE]:
        return list(command[3:])
    return None


if __name__ == "__main__":
    serve()
