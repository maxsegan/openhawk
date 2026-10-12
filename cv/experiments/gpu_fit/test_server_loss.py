"""A fit server that dies between a client's send and its recv: the client falls back to the CPU
for that evaluation, loudly (counted, warned), instead of the attempt dying of EOFError
(2026-10-01 dry run, Wang-Williams pt0193)."""

import os
import subprocess
import sys
import time

import numpy as np
import pytest

from cv.experiments.gpu_fit import gpu_fit, refine_hook, server

# Accepts one client, takes its request, then SIGKILLs itself before replying.
DYING_SERVER = """
import os, signal, sys
from multiprocessing.connection import Listener
listener = Listener(sys.argv[1], family="AF_UNIX", authkey=b"s6-gpu-fit")
connection = listener.accept()
connection.recv()
os.kill(os.getpid(), signal.SIGKILL)
"""


@pytest.fixture
def dying_server(tmp_path, monkeypatch):
    socket = tmp_path / "fit.sock"
    process = subprocess.Popen([sys.executable, "-c", DYING_SERVER, str(socket)])
    for _ in range(200):
        if socket.exists():
            break
        time.sleep(0.05)
    monkeypatch.setenv("TENNIS_S6_GPU_FIT_SERVER", str(socket))
    monkeypatch.setattr(server, "_CLIENT", None)
    monkeypatch.setattr(gpu_fit, "_ERRORS", {"cpu_fallbacks": 0, "examples": []})
    monkeypatch.delenv(gpu_fit.ERROR_LOG_ENV, raising=False)
    yield process
    process.kill()
    process.wait()


def test_exchange_raises_server_lost_and_drops_the_connection(dying_server):
    with pytest.raises(server.ServerLost) as caught:
        server.evaluate_request(os.environ["TENNIS_S6_GPU_FIT_SERVER"], {}, [[0.0]], 64, True)
    assert isinstance(caught.value, OSError)  # the hooks' existing fallback class
    assert server._CLIENT is None
    assert dying_server.wait(timeout=10) == -9


def test_gradient_evaluation_falls_back_to_the_cpu(dying_server, capsys):
    problem = {"use_term_pen": 0, "use_dir_pen": 1, "nwin": 1}

    def objective(x):
        return float(np.sum(np.asarray(x) ** 2))

    constraint = {"type": "ineq", "fun": lambda x: np.array([1.0 - x[0]])}
    options, constraints = refine_hook._gpu_gradients(
        problem, [constraint], [(-2.0, 2.0), (-2.0, 2.0)], {"eps": 1e-6}, row := {}
    )
    points = [np.array([0.5, 0.25]), np.array([0.5 + 1e-6, 0.25])]
    values = options["workers"](objective, points)  # the server dies inside this request
    assert [float(v[0]) for v in values] == [objective(p) for p in points]
    assert row["gpu_grad"]["cpu_points"] == 2
    assert gpu_fit.error_receipt()["cpu_fallbacks"] == 1
    assert "refine gradient fell back to the CPU" in capsys.readouterr().err
    # The server is gone: the constraint's map falls back too (no socket left to connect to).
    dying_server.wait(timeout=10)
    jac = constraints[0]["jac"](np.array([0.5, 0.25]))
    assert np.allclose(jac, [[-1.0, 0.0]])
    assert gpu_fit.error_receipt()["cpu_fallbacks"] == 2
