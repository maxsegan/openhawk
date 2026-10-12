from __future__ import annotations

import pytest

from cv.pipeline import player_trt, switch_fallbacks


def test_mode_defaults_to_torch_and_rejects_unknown_values(monkeypatch) -> None:
    monkeypatch.delenv("TENNIS_PLAYER_ENGINE", raising=False)
    assert player_trt.mode() == "torch"
    monkeypatch.setenv("TENNIS_PLAYER_ENGINE", "trt_fp16")
    assert player_trt.mode() == "trt_fp16"
    monkeypatch.setenv("TENNIS_PLAYER_ENGINE", "trt_int8")
    with pytest.raises(SystemExit):
        player_trt.mode()


def test_load_is_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("TENNIS_PLAYER_ENGINE", raising=False)
    assert player_trt.load("unused.pt", "cuda:0", (1088, 1920)) is None


def test_load_keeps_the_torch_model_and_records_when_tensorrt_fails(monkeypatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.setenv("TENNIS_PLAYER_ENGINE", "trt_fp16")
    recorded = []
    monkeypatch.setattr(switch_fallbacks, "record", lambda *a, **k: recorded.append((a, k)))

    def missing(*_args):
        raise ImportError("no tensorrt here")

    monkeypatch.setattr(player_trt, "engine_file", missing)
    assert player_trt.load("unused.pt", "cuda:0", (1088, 1920)) is None
    assert recorded and recorded[0][0][:2] == ("TENNIS_PLAYER_ENGINE", "tensorrt_missing")
