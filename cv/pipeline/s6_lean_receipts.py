"""Owner-approved lean S6 receipts (2026-10-01), behind ``TENNIS_S6_LEAN`` (default off).

With the switch on:

- component packets reference their four source documents by path + sha256
  (``s6_component_scope.REFERENCE_SCHEMA``) instead of embedding them;
- a component result keeps ``evaluation_context`` in a sidecar ``evaluation_context.json``
  next to ``result.json`` and records ``{"schema": SIDECAR_SCHEMA, "record": file record}``
  in its place (``final_state_binding`` still hashes the full context; across the two forms
  only its ``evaluation_context`` digest moves, because the context's attempt then carries the
  source reference instead of the embedded documents);
- S6 JSON documents are written compact (no indentation);
- the shared refinement's first-impact root runs compiled (exact; same values and errors).

``TENNIS_S6_LEAN_PARTS`` selects parts (``part``).

Readers accept both forms: the form a document carries decides how it is read, never the
switch. Rollback: unset the switch; documents already written stay readable.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

SWITCH = "TENNIS_S6_LEAN"
SIDECAR_SCHEMA = "s6_evaluation_context_sidecar_v1"
SIDECAR_NAME = "evaluation_context.json"


def enabled() -> bool:
    value = os.environ.get(SWITCH, "off")
    if value not in ("on", "off", "1", "0"):
        raise ValueError(f"{SWITCH} must be on or off")
    return value in ("on", "1")


PARTS = "TENNIS_S6_LEAN_PARTS"
ALL_PARTS = ("sources", "sidecar", "native")


def part(name: str) -> bool:
    """One part of the switch (``TENNIS_S6_LEAN_PARTS``, comma list; default all parts):
    ``sources`` (references + once-per-process verification), ``sidecar`` (evaluation context
    sidecar + compact JSON), ``native`` (compiled first-impact root). For per-step measurement
    and partial rollback."""
    if name not in ALL_PARTS:
        raise ValueError(f"unknown lean part {name!r}")
    if not enabled():
        return False
    chosen = os.environ.get(PARTS, ",".join(ALL_PARTS)).split(",")
    if set(chosen) - set(ALL_PARTS):
        raise ValueError(f"{PARTS} must name parts of {ALL_PARTS}")
    return name in chosen


def indent() -> int | None:
    """Indentation for S6 JSON documents: 2 (historical) or None (compact, lean)."""
    return None if part("sidecar") else 2


def is_sidecar(value) -> bool:
    return isinstance(value, dict) and value.get("schema") == SIDECAR_SCHEMA


def _resolve(record: dict) -> Path:
    from cv.pipeline import paths

    base = record.get("path_base")
    if base == "repository":
        return paths.REPO_ROOT / record["path"]
    if base == "TENNIS_DATA_ROOT":
        return paths.data_root() / record["path"]
    raise ValueError("evaluation context sidecar requires a repository or TENNIS_DATA_ROOT path")


def evaluation_context(result: dict, beside: Path | None = None) -> dict | None:
    """A result's evaluation context, read from its sidecar when the result references one.

    ``beside``: the directory the result was read from (its own folder, or an archive
    extraction of it); the sidecar is looked for there first, then at its recorded path. The
    sidecar's bytes must match the recorded sha256. Missing context returns None."""
    value = result.get("evaluation_context")
    if not is_sidecar(value):
        return value
    local = None if beside is None else Path(beside) / SIDECAR_NAME
    path = local if local is not None and local.is_file() else _resolve(value["record"])
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != value["record"]["sha256"]:
        raise ValueError("evaluation context sidecar changed")
    return json.loads(data)


def inline(result: dict, beside: Path | None = None) -> dict:
    """A shallow copy of ``result`` with a referenced evaluation context read back in."""
    if not is_sidecar(result.get("evaluation_context")):
        return result
    return result | {"evaluation_context": evaluation_context(result, beside)}


def write_sidecar(output: Path, context: dict) -> dict:
    """Write ``context`` next to a component result and return the reference that replaces it."""
    from cv.experiments.connected_shooting.search_budget import json_default
    from cv.pipeline import provenance

    path = output / SIDECAR_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(context, default=json_default, allow_nan=False) + "\n")
    temporary.replace(path)
    record = provenance.file_record(path)
    return {
        "schema": SIDECAR_SCHEMA,
        "record": {key: record[key] for key in ("path_base", "path", "sha256", "bytes")},
    }
