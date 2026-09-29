import pytest

from cv.pipeline.vlm_policy import (
    configured_vlm_model,
    reasoning_request_options,
    validate_vlm_model,
)


@pytest.mark.parametrize(
    "model",
    (
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "Qwen/Qwen2_5-VL-72B-Instruct",
        "Qwen2.5-VL-7B",
    ),
)
def test_obsolete_unregistered_models_do_not_become_defaults(model):
    with pytest.raises(ValueError, match="evaluate and register"):
        validate_vlm_model(model)


def test_small_reasoning_model_is_rejected():
    with pytest.raises(ValueError, match="reasoning-model floor"):
        validate_vlm_model("Qwen/Qwen3-VL-8B-Instruct", minimum_billions=32, require_tracked=False)


def test_current_tracked_model_is_allowed():
    assert validate_vlm_model("Qwen/Qwen3.6-35B-A3B").endswith("35B-A3B")


def test_untracked_model_is_rejected():
    with pytest.raises(ValueError, match="evaluate and register"):
        validate_vlm_model("qwen3.7-vl")


def test_model_configuration_is_required(monkeypatch):
    monkeypatch.delenv("VLM_MODEL", raising=False)
    with pytest.raises(ValueError, match="must name"):
        configured_vlm_model()


def test_reasoning_request_options_are_model_specific_and_bounded():
    assert reasoning_request_options("Qwen/Qwen3.5-397B-A17B", "medium", answer_tokens=300) == {
        "chat_template_kwargs": {"enable_thinking": True},
        "max_tokens": 1324,
    }
    assert reasoning_request_options("thinkingmachines/Inkling-Small", "low", answer_tokens=80) == {
        "chat_template_kwargs": {"reasoning_effort": "low"},
        "max_tokens": 464,
    }


def test_reasoning_request_options_accept_explicit_large_cap(monkeypatch):
    monkeypatch.setenv("VLM_REASONING_MAX_TOKENS", "24000")
    assert (
        reasoning_request_options("Qwen/Qwen3.5-397B-A17B", "medium", answer_tokens=300)[
            "max_tokens"
        ]
        == 24000
    )


def test_explicit_unregistered_probe_is_not_a_version_ban():
    model = "Qwen/Qwen2.5-VL-72B-Instruct"
    assert validate_vlm_model(model, require_tracked=False) == model
