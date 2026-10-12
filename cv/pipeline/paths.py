"""Portable repository, data, and model-cache path resolution."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _configured(name: str) -> str | None:
    if value := os.environ.get(name):
        return value
    dotenv = REPO_ROOT / ".env"
    if not dotenv.exists():
        return None
    for raw_line in dotenv.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip().strip("\"'")
    return None


def data_root() -> Path:
    """Return the configured shared data root, or the checkout's linked ``data`` tree."""
    configured = _configured("TENNIS_DATA_ROOT")
    return Path(configured).expanduser().resolve() if configured else REPO_ROOT / "data"


def raw_root() -> Path:
    return data_root() / "raw"


def processed_root() -> Path:
    return data_root() / "processed"


def tracker_root() -> Path:
    configured = _configured("TENNIS_TRACKER_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "tennis-trackers" / "WASB-SBDT"


def resolve_data_path(value: str | Path) -> Path:
    """Resolve a manifest path relative to the data root.

    ``data/...`` remains accepted for older versioned manifests. New manifests should store paths
    below ``TENNIS_DATA_ROOT`` without a machine-specific prefix.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "data":
        return REPO_ROOT / path
    return data_root() / path


def data_relative(path: str | Path) -> str:
    """Encode a path below the configured data root for a portable manifest."""
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(data_root().resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"path is outside TENNIS_DATA_ROOT: {resolved}") from error


def require_paths(paths: dict[str, Path]) -> None:
    missing = {name: str(path) for name, path in paths.items() if not path.exists()}
    if missing:
        rendered = ", ".join(f"{name}={path}" for name, path in sorted(missing.items()))
        raise FileNotFoundError(
            f"missing external artifacts ({rendered}); configure TENNIS_DATA_ROOT or "
            "TENNIS_TRACKER_ROOT and run scripts/shared_data.py link"
        )
