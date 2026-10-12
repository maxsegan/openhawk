"""The GPU refine hook falls back to the unchanged SLSQP when the GPU cannot be used."""

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from cv.experiments.gpu_fit import gpu_fit, refine_hook

CAPTURE = Path("data/processed/gpu-fit-20260930/cap1/refine/1976090_0000.pkl")


@pytest.mark.skipif(not CAPTURE.is_file(), reason="captured refine problem not available")
def test_unusable_device_falls_back_to_cpu(monkeypatch, tmp_path):
    record = pickle.loads(CAPTURE.read_bytes())
    cpu = gpu_fit.probe(record)
    names = ("scene", "initial", "bounces", "native", "axes", "duration", "maxiter")
    call = dict(zip(names, record["args"])) | dict(record["kwargs"])
    monkeypatch.setenv("TENNIS_S6_GPU_FIT", "on")
    monkeypatch.setenv("TENNIS_S6_GPU_FIT_DEVICE", "99")
    monkeypatch.setenv("TENNIS_S6_GPU_FIT_LOG", str(tmp_path))
    result = refine_hook.solver(call)(
        cpu.cost,
        cpu.x0,
        method="SLSQP",
        bounds=list(zip(cpu.lo, cpu.hi)),
        constraints=cpu.constraints,
        options=cpu.options,
    )
    assert np.isfinite(result.fun) and result.nit >= 1
    row = next(tmp_path.glob("*.jsonl")).read_text()
    assert '"route": "cpu"' in row and "gpu_error" in row


@pytest.mark.skipif(not CAPTURE.is_file(), reason="captured refine problem not available")
def test_launches_follow_the_device_variable(monkeypatch):
    """Diagnostics (parity.compare -> evaluate) must not fall back to card 0."""
    record = pickle.loads(CAPTURE.read_bytes())
    problem = gpu_fit.pack(record, gpu_fit.probe(record))
    monkeypatch.setenv("TENNIS_S6_GPU_FIT_DEVICE", "99")
    with pytest.raises(RuntimeError):
        gpu_fit.evaluate(problem, np.zeros((1, gpu_fit.MAX_P)))


FIXED = Path(
    "data/processed/gpu-fit/fixtures/refine_v1/netcases__2113730_0003.pkl"
)


@pytest.mark.skipif(not FIXED.is_file(), reason="parity fixture not available")
@pytest.mark.skipif(
    "TENNIS_S6_GPU_FIT_DEVICE" not in os.environ,
    reason="needs an explicit GPU (TENNIS_S6_GPU_FIT_DEVICE)",
)
def test_fused_gradient_launch_is_bit_identical(monkeypatch, tmp_path):
    """One launch per SLSQP iterate returns exactly the unfused polish (fixture has a fixed variable)."""
    record = pickle.loads(FIXED.read_bytes())
    cpu = gpu_fit.probe(record)
    assert np.any(cpu.lo == cpu.hi)
    names = ("scene", "initial", "bounces", "native", "axes", "duration", "maxiter")
    call = dict(zip(names, record["args"])) | dict(record["kwargs"])
    monkeypatch.setenv("TENNIS_S6_GPU_FIT", "on")
    monkeypatch.setenv("TENNIS_S6_GPU_FIT_POLISH", "all")
    answers = {}
    for fuse in ("off", "on"):
        monkeypatch.setenv("TENNIS_S6_GPU_FIT_GRAD_FUSE", fuse)
        monkeypatch.setenv("TENNIS_S6_GPU_FIT_LOG", str(tmp_path / fuse))
        result = refine_hook.solver(call)(
            cpu.cost,
            cpu.x0,
            method="SLSQP",
            bounds=list(zip(cpu.lo, cpu.hi)),
            constraints=cpu.constraints,
            options=cpu.options,
        )
        row = json.loads(next((tmp_path / fuse).glob("*.jsonl")).read_text())
        assert "gpu_grad" in row, json.dumps(row)
        answers[fuse] = (result.x, row["gpu_grad"]["launches"])
    assert np.array_equal(answers["on"][0], answers["off"][0])
    assert answers["on"][1] < answers["off"][1]
