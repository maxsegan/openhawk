"""A GPU-fit failure answered on the CPU is counted, warned and logged for the step receipt."""

import json

from cv.experiments.gpu_fit import gpu_fit
from cv.pipeline import s6_broadcast_backend as backend


def test_fallback_is_counted_warned_and_logged(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gpu_fit, "_ERRORS", {"cpu_fallbacks": 0, "examples": []})
    with backend.gpu_fit_error_log(tmp_path) as fallbacks:
        gpu_fit.record_error("refine search", RuntimeError("out of memory"))
        gpu_fit.record_error("seed solve", RuntimeError("out of memory"))
        assert fallbacks() == 2
    assert gpu_fit.ERROR_LOG_ENV not in __import__("os").environ
    assert "fell back to the CPU" in capsys.readouterr().err
    assert gpu_fit.error_receipt()["cpu_fallbacks"] == 2
    lines = (tmp_path / backend.GPU_FIT_ERROR_LOG).read_text().splitlines()
    assert json.loads(lines[0])["where"] == "refine search"
    assert backend.gpu_fit_errors(tmp_path / "elsewhere") == 0
