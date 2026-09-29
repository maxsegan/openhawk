"""Hash-pinned pointer to the owner-approved current-standard event truth."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cv.pipeline.paths import processed_root


CONFIG_PATH = Path(__file__).with_name("current_standard_event_truth.json")


def load_config() -> dict:
    document = json.loads(CONFIG_PATH.read_text())
    if document.get("schema") != "current_standard_event_truth_pointer_v1":
        raise ValueError(f"unsupported current-standard truth config: {CONFIG_PATH}")
    return document


def _verified_path(role: str) -> Path:
    record = load_config()[role]
    path = processed_root() / record["relative_to_processed_root"]
    if not path.is_file():
        raise FileNotFoundError(f"missing {role} event truth: {path}")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != record["sha256"]:
        raise ValueError(
            f"{role} event truth hash mismatch: expected {record['sha256']}, observed {observed}"
        )
    return path


def authoritative_truth_path() -> Path:
    return _verified_path("authoritative")


def archived_v1_truth_path() -> Path:
    return _verified_path("archived_v1")
