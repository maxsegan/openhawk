"""Fail-closed provenance records for automatic pipeline runs."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future
import hashlib
import threading
import time
import json
import subprocess
import os
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "pipeline_provenance_v1"
AUTOMATIC_MODE = "automatic"
DIAGNOSTIC_MODE = "diagnostic"

FORBIDDEN_CONFIG_KEYS = {
    "agent_annotations_loaded",
    "allow_manual_anchors",
    "allow_reviewed_play_clusters",
    "labels_loaded",
    "labels",
    "manual_court_anchors_allowed",
    "match_specific_overrides_allowed",
    "owner_truth_loaded",
    "truth",
    "play_clusters",
    "preselected_match_timestamps_loaded",
    "rep_frames",
    "reviewed_play_clusters_loaded",
    "selected_segments",
}

REQUIRED_FIELDS = {
    "schema",
    "mode",
    "git",
    "source_videos",
    "models",
    "configuration",
    "reused_artifacts",
    "fallbacks",
    "human_inputs",
    "reviewed_inputs",
    "manual_overrides",
}


class ProvenanceError(RuntimeError):
    """Raised when an automatic run contains forbidden or incomplete provenance."""


_HASH_CACHE_LIMIT = 65536
_HASH_SETTLE_NS = 1_000_000_000
_HASH_CACHE: OrderedDict[tuple, str] = OrderedDict()
_HASH_PENDING: dict[tuple, Future] = {}
_HASH_LOCK = threading.Lock()


def _reset_hash_cache_after_fork() -> None:
    # Parent threads and unfinished Futures do not exist in the forked child.
    # Replace the lock too: acquiring an inherited locked mutex could deadlock.
    global _HASH_CACHE, _HASH_PENDING, _HASH_LOCK
    _HASH_CACHE = OrderedDict()
    _HASH_PENDING = {}
    _HASH_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_hash_cache_after_fork)


def _file_identity(path: Path) -> tuple:
    stat = path.stat()
    return (
        str(path),
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_nlink,
        stat.st_ctime_ns,
    )


def _link_metadata_change(before: tuple, after: tuple) -> bool:
    # ctime alone is not evidence of harmlessness: a writer can restore mtime.
    # Require the same path/inode/size/mtime and an actual link-count change.
    return before[:5] == after[:5] and before[5] != after[5]


def _verify_hash_identity(path: Path, identity: tuple, digest: str, *, context: str) -> tuple:
    """Re-read ctime/link-count races; never trust metadata to prove equal bytes."""
    for _ in range(3):
        current = _file_identity(path)
        if current == identity:
            return current
        if not _link_metadata_change(identity, current):
            raise ProvenanceError(f"file changed {context}: {path.name}")
        # Export hardlinks change ctime while readers hash the original image.
        # A second full digest must agree, and that read must become stable.
        # This also rejects a same-size writer that restores mtime while a link
        # happens to be created: unchanged structural metadata is insufficient.
        if _read_sha256(path, current) != digest:
            raise ProvenanceError(f"file changed {context}: {path.name}")
        identity = current
    if _file_identity(path) != identity:
        raise ProvenanceError(f"file changed repeatedly {context}: {path.name}")
    return identity


def _read_sha256(path: Path, identity: tuple) -> str:
    # Callers must verify the final identity/digest after tolerated link metadata changes.
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        stat = os.fstat(handle.fileno())
        opened = (
            str(path),
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_nlink,
            stat.st_ctime_ns,
        )
        if opened != identity and not _link_metadata_change(identity, opened):
            raise ProvenanceError(f"file changed before hashing: {path.name}")
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """Hash unchanged files once per bounded process cache, coalescing concurrent reads.

    Cache keys include inode and ctime as well as size/mtime: replacing a file or
    restoring its mtime after editing invalidates the old digest. Files with
    ctime less than one second old are not cached, avoiding timestamp coalescing
    during rapid writes. Every call checks identity before and after lookup.
    A changed link count with otherwise stable file metadata requires a second
    identical full read; ordinary writes/replacements still fail closed.
    """
    path = Path(path).expanduser().resolve()
    identity = _file_identity(path)
    with _HASH_LOCK:
        cached = _HASH_CACHE.get(identity)
        if cached is not None:
            _HASH_CACHE.move_to_end(identity)
            pending, owner = None, False
        else:
            pending = _HASH_PENDING.get(identity)
            owner = pending is None
            if owner:
                pending = Future()
                _HASH_PENDING[identity] = pending
    if cached is not None:
        digest = cached
    elif not owner:
        digest = pending.result()
    else:
        try:
            digest = _read_sha256(path, identity)
            verified = _verify_hash_identity(path, identity, digest, context="while hashing")
            with _HASH_LOCK:
                # Filesystem timestamps can coalesce rapid same-size writes.
                # Recently changed files remain cheap ordinary full reads until
                # their metadata is older than the conservative clock window.
                if time.time_ns() - verified[-1] >= _HASH_SETTLE_NS:
                    _HASH_CACHE[verified] = digest
                    _HASH_CACHE.move_to_end(verified)
                    while len(_HASH_CACHE) > _HASH_CACHE_LIMIT:
                        _HASH_CACHE.popitem(last=False)
                _HASH_PENDING.pop(identity)
                pending.set_result(digest)
            identity = verified
        except BaseException as error:
            with _HASH_LOCK:
                _HASH_PENDING.pop(identity, None)
                pending.set_exception(error)
            raise
    _verify_hash_identity(path, identity, digest, context="during digest lookup")
    return digest


def portable_path(path: str | Path) -> dict[str, str]:
    """Describe a path without embedding the current machine's home or checkout location."""
    from cv.pipeline.paths import REPO_ROOT, data_root, tracker_root

    resolved = Path(path).expanduser().resolve()
    roots = (
        ("TENNIS_DATA_ROOT", data_root().resolve()),
        ("TENNIS_TRACKER_ROOT", tracker_root().resolve()),
        ("repository", REPO_ROOT.resolve()),
    )
    for base, root in roots:
        try:
            return {"path": resolved.relative_to(root).as_posix(), "path_base": base}
        except ValueError:
            continue
    return {"path": resolved.name, "path_base": "unconfigured_external"}


def file_record(path: str | Path, *, role: str | None = None) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    record: dict[str, Any] = {
        **portable_path(resolved),
        "sha256": file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }
    if role is not None:
        record["role"] = role
    return record


def portable_configuration(document: Any) -> Any:
    """Replace absolute configuration paths with base-qualified portable references."""
    if isinstance(document, dict):
        return {key: portable_configuration(value) for key, value in document.items()}
    if isinstance(document, list):
        return [portable_configuration(value) for value in document]
    if isinstance(document, tuple):
        return [portable_configuration(value) for value in document]
    if isinstance(document, os.PathLike):
        document = os.fspath(document)
    if isinstance(document, str) and Path(document).expanduser().is_absolute():
        return portable_path(document)
    return document


def git_record(root: str | Path) -> dict[str, Any]:
    root = Path(root)

    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _truthy_forbidden_values(document: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(document, dict):
        for key, value in document.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            lowered = str(key).lower()
            human_named = (
                "manual" in lowered
                or "reviewed" in lowered
                or "owner_truth" in lowered
                or "agent_annotation" in lowered
                or lowered in {"label", "labels", "truth"}
            )
            if (key in FORBIDDEN_CONFIG_KEYS or human_named) and bool(value):
                found.append(path)
            found.extend(_truthy_forbidden_values(value, path))
    elif isinstance(document, list):
        for index, value in enumerate(document):
            found.extend(_truthy_forbidden_values(value, f"{prefix}[{index}]"))
    return found


def build_provenance(
    *,
    root: str | Path,
    mode: str,
    source_videos: Iterable[dict[str, Any]] = (),
    models: Iterable[dict[str, Any]] = (),
    configuration: dict[str, Any] | None = None,
    reused_artifacts: Iterable[dict[str, Any]] = (),
    fallbacks: Iterable[dict[str, Any]] = (),
    human_inputs: Iterable[dict[str, Any]] = (),
    reviewed_inputs: Iterable[dict[str, Any]] = (),
    manual_overrides: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    document = {
        "schema": SCHEMA,
        "mode": mode,
        "git": git_record(root),
        "source_videos": list(source_videos),
        "models": list(models),
        "configuration": portable_configuration(configuration or {}),
        "reused_artifacts": list(reused_artifacts),
        "fallbacks": list(fallbacks),
        "human_inputs": list(human_inputs),
        "reviewed_inputs": list(reviewed_inputs),
        "manual_overrides": list(manual_overrides),
    }
    validate_provenance(document)
    return document


def validate_provenance(document: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_FIELDS - set(document))
    if missing:
        raise ProvenanceError(f"provenance missing required fields: {missing}")
    if document["schema"] != SCHEMA:
        raise ProvenanceError(f"unsupported provenance schema: {document['schema']!r}")
    if document["mode"] not in {AUTOMATIC_MODE, DIAGNOSTIC_MODE}:
        raise ProvenanceError(f"invalid provenance mode: {document['mode']!r}")
    for field in ("source_videos", "models", "reused_artifacts"):
        for index, record in enumerate(document[field]):
            if not record.get("path") and not record.get("name"):
                raise ProvenanceError(f"{field}[{index}] lacks a path or model name")
            if record.get("path") and not record.get("sha256"):
                raise ProvenanceError(f"{field}[{index}] path lacks sha256")
    for field in ("human_inputs", "reviewed_inputs", "manual_overrides"):
        for index, record in enumerate(document[field]):
            if not record.get("type") and not record.get("role"):
                raise ProvenanceError(f"{field}[{index}] lacks a type or role")
            if record.get("path") and not record.get("sha256"):
                raise ProvenanceError(f"{field}[{index}] path lacks sha256")
    if document["mode"] == AUTOMATIC_MODE:
        forbidden_lists = {
            field: document[field]
            for field in ("human_inputs", "reviewed_inputs", "manual_overrides")
            if document[field]
        }
        forbidden_config = _truthy_forbidden_values(document["configuration"])
        if forbidden_lists or forbidden_config:
            raise ProvenanceError(
                "automatic provenance contains forbidden inputs: "
                f"lists={sorted(forbidden_lists)}, config={forbidden_config}"
            )


def write_provenance(path: str | Path, document: dict[str, Any]) -> Path:
    validate_provenance(document)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


def load_provenance(path: str | Path, *, require_automatic: bool = False) -> dict[str, Any]:
    path = Path(path)
    document = json.loads(path.read_text())
    validate_provenance(document)
    if require_automatic and document["mode"] != AUTOMATIC_MODE:
        raise ProvenanceError(f"automatic benchmark received {document['mode']} provenance: {path}")
    return document


def assert_automatic_document(document: dict[str, Any], *, context: str) -> None:
    forbidden = _truthy_forbidden_values(document)
    if forbidden:
        raise ProvenanceError(f"{context} contains forbidden automatic inputs: {forbidden}")
