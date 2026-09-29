"""Repository-wide policy for selecting vision-language models."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

MODEL_REGISTRY = Path(__file__).with_name("vlm_models.json")
REASONING_LEVELS = ("none", "minimal", "low", "medium", "high")
REASONING_TOKEN_BUDGETS = {
    "none": 0,
    "minimal": 128,
    "low": 384,
    "medium": 1024,
    "high": 2048,
}


@lru_cache(maxsize=1)
def tracked_vlm_models() -> dict[str, dict]:
    document = json.loads(MODEL_REGISTRY.read_text())
    return {row["id"]: row for row in document["models"]}


def validate_vlm_model(
    model: str,
    *,
    minimum_billions: float = 0.0,
    require_tracked: bool = True,
) -> str:
    """Require explicit model identity, registration and any caller-specified size floor."""
    name = model.strip()
    if not name:
        raise ValueError("VLM_MODEL must name an explicitly configured current model")
    size = re.search(r"(?:^|[-_/])(\d+(?:\.\d+)?)b(?:$|[-_/])", name, re.IGNORECASE)
    if size is not None and float(size.group(1)) < minimum_billions:
        raise ValueError(f"VLM {name!r} is below the {minimum_billions:g}B reasoning-model floor")
    if require_tracked and name not in tracked_vlm_models():
        raise ValueError(
            f"VLM {name!r} is not in {MODEL_REGISTRY.name}; evaluate and register the model before integration"
        )
    return name


def configured_vlm_model(*, minimum_billions: float = 0.0) -> str:
    return validate_vlm_model(
        os.environ.get("VLM_MODEL", ""),
        minimum_billions=minimum_billions,
    )


def reasoning_request_options(
    model: str,
    level: str,
    *,
    answer_tokens: int,
) -> dict:
    """Return bounded, model-specific thinking controls for one request."""
    name = validate_vlm_model(model)
    if level not in REASONING_TOKEN_BUDGETS:
        raise ValueError(f"unknown reasoning level {level!r}; choose from {REASONING_LEVELS}")
    if answer_tokens <= 0:
        raise ValueError("answer_tokens must be positive")
    budget = REASONING_TOKEN_BUDGETS[level]
    configured_cap = int(os.environ.get("VLM_REASONING_MAX_TOKENS", "0") or 0)
    if name == "thinkingmachines/Inkling-Small":
        template_kwargs = {"reasoning_effort": level}
    else:
        template_kwargs = {"enable_thinking": level != "none"}
    return {
        "chat_template_kwargs": template_kwargs,
        "max_tokens": max(answer_tokens + budget, configured_cap),
    }
