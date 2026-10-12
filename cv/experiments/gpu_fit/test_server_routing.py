"""With the fit server set, a client never opens a CUDA context: solve and evaluate both run
on the server's card, bit for bit as they run locally."""

import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(os.environ.get("TENNIS_DATA_ROOT", "data"))
FIXTURE = sorted(glob.glob(str(ROOT / "processed/gpu-fit/fixtures/refine_v1/*.pkl")))[:1]
REPO = Path(__file__).resolve().parents[3]

CLIENT = """
import pickle, sys
import numpy as np
from cv.experiments.gpu_fit import gpu_fit
record = pickle.load(open(sys.argv[1], "rb"))
cpu = gpu_fit.probe(record); problem = gpu_fit.pack(record, cpu)
seeds = gpu_fit.multistart(cpu.x0, cpu.lo, cpu.hi, 4, np.random.default_rng(0))
results, _, _ = gpu_fit.solve(np.atleast_1d(problem), np.zeros(4, np.int32), seeds, max_iter=20, max_iter64=3, report64=True)
P = int(problem["P"]); qs = np.vstack([np.asarray(seeds)[:, :P], results["q"][:, :P]])
out = {"q": results["q"], "cost": results["cost_final"]}
for precision in (64, 32):
    out[f"rows{precision}"], out[f"costs{precision}"], out[f"ok{precision}"] = gpu_fit.evaluate(problem, qs, precision=precision)
out["raw"] = gpu_fit.evaluate(problem, qs, raw=True)[3]
np.savez(sys.argv[2], **out)
"""


@pytest.mark.skipif(
    not FIXTURE or "TENNIS_S6_GPU_FIT_DEVICE" not in os.environ,
    reason="needs a refine_v1 fixture and an explicit GPU (TENNIS_S6_GPU_FIT_DEVICE)",
)
def test_client_without_any_cuda_device_gets_bit_identical_results_via_server(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    socket = tmp_path / "fit.sock"
    server = subprocess.Popen(
        [sys.executable, "-m", "cv.experiments.gpu_fit.server", "--socket", str(socket)],
        env=env,
        cwd=REPO,
    )
    try:
        for _ in range(120):
            if socket.exists():
                break
            time.sleep(0.5)
        script = tmp_path / "client.py"
        script.write_text(CLIENT)
        client_env = {**env, "CUDA_VISIBLE_DEVICES": "", "TENNIS_S6_GPU_FIT_SERVER": str(socket)}
        routed = tmp_path / "routed.npz"
        subprocess.run(
            [sys.executable, str(script), FIXTURE[0], str(routed)],
            env=client_env,
            cwd=REPO,
            check=True,
        )
        local_env = {k: v for k, v in env.items() if k != "TENNIS_S6_GPU_FIT_SERVER"}
        local = tmp_path / "local.npz"
        subprocess.run(
            [sys.executable, str(script), FIXTURE[0], str(local)],
            env=local_env,
            cwd=REPO,
            check=True,
        )
    finally:
        server.terminate()
        server.wait(timeout=30)
    a, b = np.load(routed), np.load(local)
    assert sorted(a.files) == sorted(b.files)
    for name in a.files:
        assert a[name].tobytes() == b[name].tobytes(), name
