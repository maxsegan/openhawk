from __future__ import annotations

import pytest

from cv.pipeline import ball_trt


def _heat(torch, seed: int = 1):
    class Heat(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            torch.manual_seed(seed)
            self.conv = torch.nn.Conv2d(9, 8, 3, padding=1)
            self.norm = torch.nn.BatchNorm2d(8)
            self.head = torch.nn.Conv2d(8, 3, 1)

        def forward(self, x):
            return (self.head(torch.relu(self.norm(self.conv(x)))),)

    return Heat().eval()


def test_mode_defaults_to_torch_and_rejects_unknown_values(monkeypatch) -> None:
    monkeypatch.delenv("TENNIS_BALL_ENGINE", raising=False)
    assert ball_trt.mode() == "torch"
    monkeypatch.setenv("TENNIS_BALL_ENGINE", "trt_fp16")
    assert ball_trt.mode() == "trt_fp16"
    monkeypatch.setenv("TENNIS_BALL_ENGINE", "trt_int8")
    with pytest.raises(SystemExit):
        ball_trt.mode()


def test_wrap_leaves_the_torch_model_alone_when_off(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("TENNIS_BALL_ENGINE", raising=False)
    model = _heat(torch)
    assert ball_trt.wrap(model, "wasb", "unused", 0) is model
    assert not ball_trt.is_half(model)
    assert ball_trt.identity(model) == {"engine": "torch"}


def test_wrap_falls_back_to_torch_fp16_without_tensorrt(monkeypatch, tmp_path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    weights = tmp_path / "weights.pth.tar"
    weights.write_bytes(b"weights")
    monkeypatch.setenv("TENNIS_BALL_ENGINE", "trt_fp16")
    monkeypatch.setenv("TENNIS_TRT_ENGINE_DIR", str(tmp_path / "engines"))

    def missing():
        raise ImportError("no TensorRT here")

    monkeypatch.setattr(ball_trt, "tensorrt", missing)
    model = ball_trt.wrap(_heat(torch).cuda(), "wasb", str(weights), 0, (16, 32))
    assert ball_trt.is_half(model)
    assert next(model.parameters()).dtype == torch.float16
    assert ball_trt.identity(model)["engine"].startswith("torch_fp16 (fallback: ImportError")


def test_engine_matches_torch_fp16_and_is_cached(monkeypatch, tmp_path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    try:
        ball_trt.tensorrt()
    except ImportError as error:
        pytest.skip(str(error))
    weights = tmp_path / "weights.pth.tar"
    weights.write_bytes(b"weights")
    monkeypatch.setenv("TENNIS_BALL_ENGINE", "trt_fp16")
    monkeypatch.setenv("TENNIS_TRT_ENGINE_DIR", str(tmp_path / "engines"))
    model = _heat(torch).cuda()
    reference_state = {k: v.clone() for k, v in model.state_dict().items()}
    engine = ball_trt.wrap(model, "wasb", str(weights), 0, (16, 32))
    assert isinstance(engine, ball_trt.TrtModel)
    # The export leaves the torch model exactly as it was, in eval mode.
    assert not any(module.training for module in model.modules())
    assert all(
        torch.equal(value, reference_state[key]) and value.dtype == reference_state[key].dtype
        for key, value in model.state_dict().items()
    )
    # A batch above the engine maximum runs in slices; any float input dtype is accepted.
    x = torch.randn(ball_trt.MAX_BATCH + 6, 9, 16, 32, device="cuda")
    with torch.inference_mode():
        got = engine(x)[0]
        want = model.half()(x.half())[0]
    assert got.dtype == torch.float16 and got.shape == want.shape
    assert torch.allclose(got.float(), want.float(), atol=2e-2, rtol=2e-2)
    identity = ball_trt.identity(engine)
    assert identity["engine"] == "trt_fp16" and identity["input"] == [9, 16, 32]
    built = sorted(p.name for p in (tmp_path / "engines").glob("*.engine"))
    assert len(built) == 1
    again = ball_trt.wrap(_heat(torch).cuda(), "wasb", str(weights), 0, (16, 32))
    assert again.path == engine.path


def test_shared_engines_key_drops_the_card_name(monkeypatch, tmp_path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    weights = tmp_path / "weights.pth.tar"
    weights.write_bytes(b"weights")
    monkeypatch.delenv(ball_trt.SHARED_ENGINES_ENV, raising=False)
    per_card = ball_trt.engine_key("wasb", str(weights), 0, (16, 32))
    monkeypatch.setenv(ball_trt.SHARED_ENGINES_ENV, "on")
    shared = ball_trt.engine_key("wasb", str(weights), 0, (16, 32))
    assert per_card["gpu"] == torch.cuda.get_device_name(0) and "gpu" not in shared
    assert {k: v for k, v in per_card.items() if k != "gpu"} == shared
    # Separate files: turning the switch on never reuses a per-card engine.
    assert ball_trt._engine_path(per_card) != ball_trt._engine_path(shared)
    engine = tmp_path / "x.engine"
    engine.write_bytes(b"engine")
    identity = ball_trt.engine_file_identity(engine, 0)
    assert identity["engine_sha256"] == ball_trt._sha256(str(engine))
    assert identity["ran_on"] == torch.cuda.get_device_name(0)
