"""Content-addressed stage receipts for safe pipeline resume."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterable

from cv.pipeline.provenance import (
    bulk_sha256,
    _HASH_SETTLE_NS,
    file_sha256,
    hash_threads,
    git_record,
    portable_configuration,
    portable_path,
)

SCHEMA = "pipeline_stage_receipt_v3"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _digest_json(document: Any) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def path_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_file():
        return {
            **portable_path(resolved),
            "kind": "file",
            "sha256": file_sha256(resolved),
            "bytes": resolved.stat().st_size,
        }
    if resolved.is_dir():
        key = _directory_identity(resolved)
        with _DIRECTORY_LOCK:
            cached = _DIRECTORY_RECORDS.get(str(resolved))
        if key is not None and cached is not None and cached[0] == key:
            return dict(cached[1])
        files = []
        total_bytes = 0
        digest = hashlib.sha256()
        children = sorted(candidate for candidate in resolved.rglob("*") if candidate.is_file())
        # TENNIS_HASH_THREADS: hash ahead on threads; the loop below then reads the cache.
        if hash_threads() > 1:
            bulk_sha256(children)
        for child in children:
            relative = child.relative_to(resolved).as_posix()
            size = child.stat().st_size
            child_digest = file_sha256(child)
            digest.update(f"{relative}\0{size}\0{child_digest}\n".encode())
            files.append(relative)
            total_bytes += size
        record = {
            **portable_path(resolved),
            "kind": "directory",
            "sha256": digest.hexdigest(),
            "bytes": total_bytes,
            "files": len(files),
        }
        if key is not None and _directory_identity(resolved) == key:
            with _DIRECTORY_LOCK:
                _DIRECTORY_RECORDS[str(resolved)] = (key, dict(record))
                while len(_DIRECTORY_RECORDS) > _DIRECTORY_RECORDS_LIMIT:
                    _DIRECTORY_RECORDS.pop(next(iter(_DIRECTORY_RECORDS)))
        return record
    raise FileNotFoundError(resolved)


def listing_digest(rows) -> str:
    """:func:`path_record`'s directory sha256 from (relative name, bytes, sha256) rows."""
    digest = hashlib.sha256()
    # path_record's order: sorted paths (by parts, not by string).
    for relative, size, sha in sorted(rows, key=lambda row: Path(row[0]).parts):
        digest.update(f"{relative}\0{size}\0{sha}\n".encode())
    return digest.hexdigest()


#: A stored listing of a directory's files, kept beside it as ``<directory>.listing.tsv``.
LISTING_SUFFIX = ".listing.tsv"


def directory_matches(directory: str | Path, binding: dict[str, Any]) -> bool:
    """Is ``directory`` still the recorded producer output ``binding``?

    Whole: its :func:`path_record` equals the binding. Partial: a compacted match keeps the
    listing (name, bytes, sha256 per file, tab separated) of a deleted native frame folder
    beside it and regenerates only some clips' pictures. The folder still counts as the
    producer's output when the listing reproduces the recorded sha256, file count and bytes
    and every file present has its listed size and sha256: no file can be added, changed or
    renamed, and a reader of a missing file fails as it would without this check.
    """
    resolved = Path(directory).expanduser().resolve()
    if path_record(resolved) == binding:
        return True
    listing = resolved.with_name(resolved.name + LISTING_SUFFIX)
    if not listing.is_file() or binding.get("kind") != "directory":
        return False
    rows = {}
    for line in filter(None, listing.read_text().splitlines()):
        relative, size, sha = line.split("\t")
        rows[relative] = (int(size), sha)
    if (
        listing_digest((r, size, sha) for r, (size, sha) in rows.items()) != binding.get("sha256")
        or len(rows) != binding.get("files")
        or sum(size for size, _ in rows.values()) != binding.get("bytes")
        or {k: binding.get(k) for k in ("path", "path_base")} != portable_path(resolved)
    ):
        return False
    for child in resolved.rglob("*"):
        if not child.is_file():
            continue
        entry = rows.get(child.relative_to(resolved).as_posix())
        if entry is None or entry != (child.stat().st_size, file_sha256(child)):
            return False
    return True


#: Whole-directory records keyed by the stat identity of every file under the directory
#: (the same inode/size/mtime/ctime identity the per-file digest cache trusts). A preparation
#: process verifies the same ~10^5-file native frame directory once per attempt; hashing it
#: once per process and re-walking only its metadata afterwards returns identical records.
_DIRECTORY_RECORDS: dict[str, tuple[str, dict[str, Any]]] = {}
_DIRECTORY_RECORDS_LIMIT = 64
_DIRECTORY_LOCK = threading.Lock()


def _directory_identity(resolved: Path) -> str | None:
    """Digest of every file's (relative name, stat identity), or None when not memoizable.

    ``rglob`` does not descend into symlinked directories, so under a resolved directory a
    non-symlink child is its own real path and ``lstat`` is its ``stat``. Any symlink, and
    files changed within the per-file cache's settle window, disable the memo.
    """
    settled = time.time_ns() - _HASH_SETTLE_NS
    entries: list[tuple] = []
    pending = [(os.fspath(resolved), "")]
    while pending:
        directory, prefix = pending.pop()
        with os.scandir(directory) as listing:
            for entry in listing:
                if entry.is_symlink():
                    return None
                if entry.is_dir(follow_symlinks=False):
                    pending.append((entry.path, f"{prefix}{entry.name}/"))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                status = entry.stat(follow_symlinks=False)
                if status.st_ctime_ns > settled:
                    return None
                entries.append(
                    (
                        prefix + entry.name,
                        entry.path,
                        status.st_dev,
                        status.st_ino,
                        status.st_size,
                        status.st_mtime_ns,
                        status.st_nlink,
                        status.st_ctime_ns,
                    )
                )
    entries.sort()
    return hashlib.sha256(repr(entries).encode()).hexdigest()


def _repository_python_path(module_name: str) -> Path | None:
    parts = module_name.split(".")
    if not parts or parts[0].startswith("."):
        return None
    module = REPOSITORY_ROOT.joinpath(*parts).with_suffix(".py")
    if module.is_file():
        return module.resolve()
    package = REPOSITORY_ROOT.joinpath(*parts, "__init__.py")
    return package.resolve() if package.is_file() else None


def _package_for_path(path: Path) -> str:
    relative = path.resolve().relative_to(REPOSITORY_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    else:
        parts.pop()
    return ".".join(parts)


def repository_import_names(source: str, package: str) -> tuple[list[str], set[str]]:
    """Parse the shared static/dynamic dependency declarations without executing code."""
    tree = ast.parse(source)
    names: list[str] = []
    required: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "RUNTIME_MODULE_DEPENDENCIES"
            for target in node.targets
        ):
            try:
                declared = ast.literal_eval(node.value)
            except (ValueError, TypeError) as error:
                raise ValueError("nonliteral runtime dependencies") from error
            if not isinstance(declared, (tuple, list)) or not all(
                isinstance(name, str) for name in declared
            ):
                raise ValueError("invalid runtime dependencies")
            required.update(declared)
            names.extend(declared)
        elif isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                try:
                    module = importlib.util.resolve_name("." * node.level + module, package)
                except (ImportError, ValueError):
                    module = ""
            if module:
                names.append(module)
                names.extend(f"{module}.{alias.name}" for alias in node.names)
    return names, required


def _imported_repository_paths(path: Path) -> set[Path]:
    try:
        names, required = repository_import_names(path.read_text(), _package_for_path(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set()
    imported = set()
    for name in names:
        dependency = _repository_python_path(name)
        if dependency is None and "." not in name:
            sibling = path.parent / f"{name}.py"
            dependency = sibling.resolve() if sibling.is_file() else None
        if dependency is None and name in required:
            raise ValueError(f"missing runtime dependency {name} declared by {path}")
        if dependency is not None:
            imported.add(dependency)
    return imported


def repository_code_closure(entrypoints: Iterable[Path]) -> list[Path]:
    """Follow imports and explicit subprocess/dynamic-module dependencies without importing code."""
    pending = [path.resolve() for path in entrypoints]
    observed = set()
    while pending:
        path = pending.pop()
        if path in observed or not path.is_file():
            continue
        observed.add(path)
        if not path.is_relative_to(REPOSITORY_ROOT):
            continue
        pending.extend(_imported_repository_paths(path) - observed)
    return sorted(observed)


def stage_identity(
    *,
    stage: str,
    command: Iterable[str],
    inputs: Iterable[str | Path],
    upstream_receipts: Iterable[str | Path] = (),
    configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    command = list(command)
    code_paths: list[Path] = []
    for value in command[1:]:
        candidate = Path(value).expanduser()
        if candidate.suffix == ".py" and candidate.is_file():
            code_paths.append(candidate)
    if len(command) >= 3 and command[1] == "-m":
        module_path = _repository_python_path(command[2])
        if module_path is not None:
            code_paths.append(module_path)
    code = [path_record(path) for path in repository_code_closure(dict.fromkeys(code_paths))]
    document = {
        "stage": stage,
        "command": portable_configuration(command),
        "configuration": portable_configuration(configuration or {}),
        "code": code,
        "inputs": [path_record(path) for path in inputs],
        "upstream_receipts": [path_record(path) for path in upstream_receipts],
    }
    return {**document, "fingerprint": _digest_json(document)}


def receipt_path(out_dir: str | Path, stage: str) -> Path:
    return Path(out_dir) / "run_manifests" / "stage_receipts" / f"{stage}.json"


def write_stage_receipt(
    *,
    out_dir: str | Path,
    stage: str,
    command: Iterable[str],
    inputs: Iterable[str | Path],
    outputs: Iterable[str | Path],
    upstream_receipts: Iterable[str | Path] = (),
    configuration: dict[str, Any] | None = None,
    expected_identity: dict[str, Any] | None = None,
) -> Path:
    identity = stage_identity(
        stage=stage,
        command=command,
        inputs=inputs,
        upstream_receipts=upstream_receipts,
        configuration=configuration,
    )
    if expected_identity is not None and identity != expected_identity:
        raise RuntimeError(f"{stage}: inputs or code changed while building the artifact")
    document = {
        "schema": SCHEMA,
        "identity": identity,
        # Producer metadata is outside the cache fingerprint: a docs-only Git
        # commit must not invalidate unchanged code/data. New clean executions
        # retain the exact revision needed to verify historical code inputs.
        **(
            {"implementation_source": git_record(REPOSITORY_ROOT)}
            if identity["code"]
            and all(
                r.get("path_base") == "repository" and str(r.get("path", "")).endswith(".py")
                for r in identity["code"]
            )
            else {}
        ),
        "outputs": [path_record(path) for path in outputs],
    }
    path = receipt_path(out_dir, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".json.pending")
    pending.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    pending.replace(path)
    return path


def stage_receipt_matches(
    *,
    out_dir: str | Path,
    stage: str,
    command: Iterable[str],
    inputs: Iterable[str | Path],
    outputs: Iterable[str | Path],
    upstream_receipts: Iterable[str | Path] = (),
    configuration: dict[str, Any] | None = None,
) -> bool:
    path = receipt_path(out_dir, stage)
    try:
        document = json.loads(path.read_text())
        if document.get("schema") != SCHEMA:
            return False
        current_identity = stage_identity(
            stage=stage,
            command=command,
            inputs=inputs,
            upstream_receipts=upstream_receipts,
            configuration=configuration,
        )
        if document.get("identity", {}).get("fingerprint") != current_identity["fingerprint"]:
            return False
        return document.get("outputs") == [path_record(output) for output in outputs]
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


def stage_receipt_allows_dependency_narrowing(
    *,
    out_dir: str | Path,
    stage: str,
    command: Iterable[str],
    inputs: Iterable[str | Path],
    outputs: Iterable[str | Path],
    upstream_receipts: Iterable[str | Path] = (),
    configuration: dict[str, Any] | None = None,
) -> bool:
    """Validate an old receipt when only unrelated upstream dependencies were removed."""
    path = receipt_path(out_dir, stage)
    try:
        document = json.loads(path.read_text())
        if document.get("schema") != SCHEMA:
            return False
        current = stage_identity(
            stage=stage,
            command=command,
            inputs=inputs,
            upstream_receipts=upstream_receipts,
            configuration=configuration,
        )
        observed = document.get("identity", {})
        stable_fields = ("stage", "command", "configuration", "code", "inputs")
        if any(observed.get(field) != current[field] for field in stable_fields):
            return False
        old_upstream = observed.get("upstream_receipts", [])
        if any(receipt not in old_upstream for receipt in current["upstream_receipts"]):
            return False
        return document.get("outputs") == [path_record(output) for output in outputs]
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


def remove_receipt(out_dir: str | Path, stage: str) -> None:
    try:
        os.remove(receipt_path(out_dir, stage))
    except FileNotFoundError:
        pass
