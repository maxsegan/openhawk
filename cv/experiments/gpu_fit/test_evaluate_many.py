"""Merged evaluations (``evaluate_many``, the server's ``--merge-evaluate``) return every request
exactly what its own ``evaluate_local`` launch returns, bit for bit, for mixed problems,
precisions and raw/non-raw requests; and a merging multi-card server answers concurrent
clients with the same bits, its launches
running concurrently on per-thread streams."""

import glob
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(os.environ.get("TENNIS_DATA_ROOT", "data"))
FIXTURES = sorted(glob.glob(str(ROOT / "processed/gpu-fit/fixtures/refine_v1/*.pkl")))[:6]
REPO = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.skipif(
    len(FIXTURES) < 2 or "TENNIS_S6_GPU_FIT_DEVICE" not in os.environ,
    reason="needs refine_v1 fixtures and an explicit GPU (TENNIS_S6_GPU_FIT_DEVICE)",
)


def _requests():
    from cv.experiments.gpu_fit import gpu_fit

    out = []
    for k, path in enumerate(FIXTURES):
        record = pickle.load(open(path, "rb"))
        cpu = gpu_fit.probe(record)
        problem = gpu_fit.pack(record, cpu)
        rng = np.random.default_rng(k)
        P = int(problem["P"])
        qs = np.asarray(gpu_fit.multistart(cpu.x0, cpu.lo, cpu.hi, 3 + k, rng))[:, :P]
        out.append((problem, qs, k % 2 == 0))
    return out


def _same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert x.shape == y.shape and x.tobytes() == y.tobytes()


@pytest.mark.parametrize("precision", [64, 32])
def test_evaluate_many_is_bit_identical_to_one_launch_each(precision):
    from cv.experiments.gpu_fit import gpu_fit

    requests = _requests()
    merged = gpu_fit.evaluate_many(requests, precision=precision)
    for (problem, qs, raw), answer in zip(requests, merged, strict=True):
        _same(answer, gpu_fit.evaluate_local(problem, qs, precision=precision, raw=raw))


def test_merging_server_answers_concurrent_clients_bit_for_bit(tmp_path, monkeypatch):
    from cv.experiments.gpu_fit import gpu_fit, server

    socket = tmp_path / "fit.sock"
    device = os.environ["TENNIS_S6_GPU_FIT_DEVICE"]
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cv.experiments.gpu_fit.server",
            "--socket",
            str(socket),
            "--device",
            f"{device},{device}",  # 2 x 3 dispatchers sharing one queue
            "--merge-evaluate",
            "on",
            "--streams",
            "3",  # concurrent launches on per-thread streams
            "--stats-seconds",
            "0",
        ],
        env={**os.environ, "PYTHONPATH": str(REPO)},
        cwd=REPO,
    )
    try:
        for _ in range(240):
            if socket.exists():
                break
            time.sleep(0.25)
        requests = _requests() * 8
        expected = [gpu_fit.evaluate_local(p, q, raw=raw) for p, q, raw in requests]

        def client(item):  # one connection per client, as one per fit process
            problem, qs, raw = item
            from multiprocessing.connection import Client

            connection = Client(str(socket), family="AF_UNIX", authkey=server.AUTHKEY)
            message = {
                "kind": "evaluate",
                "problem": np.atleast_1d(problem),
                "qs": np.atleast_2d(qs),
                "precision": 64,
                "raw": raw,
            }
            answers = []
            for _ in range(5):
                connection.send(message)
                answers.append(connection.recv())
            connection.close()
            return answers

        with ThreadPoolExecutor(len(requests)) as pool:
            got = list(pool.map(client, requests))
    finally:
        process.terminate()
        process.wait(timeout=30)
    merged = 0
    for answers, want in zip(got, expected, strict=True):
        for answer in answers:
            _same(answer["evaluation"], want)
            merged = max(merged, answer["merged_requests"])
    assert merged > 1  # the concurrent requests did share launches
