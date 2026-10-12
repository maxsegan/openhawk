"""Declared first-attempts execution scope for the automatic broadcast runner.

The automatic point ledger is always built over the whole broadcast.  An evaluation trial
that only wants the first few attempts still has to pay for every per-point output, because
``broadcast_runner.materialize_postseg`` copies the whole ledger into the work map and the
canonical, tracking and event stages then enumerate whatever ``pt*`` directories exist.

This module owns the one artifact that makes a shortened execution declarable rather than
implicit: ``processing_scope.json`` at the processed root.  It binds the full ledger it was
cut from, the requested and selected counts, the selected original attempt identities and
their retained native windows, and the identities excluded by the scope.  Consumers that
read a full-identity ledger (the shared S6 backend, the window-camera producer) load it
before iterating, so a scoped upstream root can never be mistaken for a full one.

It is an execution receipt, not a research framework: it selects no rows by quality, reads
no labels, shortens no native window and renumbers nothing.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from cv.pipeline import provenance

SCOPE_NAME = "processing_scope.json"
WORK_MANIFEST_NAME = "processing_manifest.json"
CONVENTIONAL_MANIFEST_NAME = "manifest.json"
SCHEMA = "automatic_evaluation_processing_scope_v1"
SELECTION_RULE = (
    "first min(N, L) rows of the complete automatic point ledger in its existing "
    "deterministic chronological order; original identities, native windows and "
    "continuation metadata preserved; no label, validity, camera-quality or fitted-state "
    "input participates"
)
# Columns copied into the scope's per-row window identity.  They are the declared native
# window and the score-point grouping, which is what a reader needs to see that nothing was
# shortened or regrouped; the ledger itself remains the authority.
WINDOW_FIELDS = (
    "pt",
    "rally_t_start",
    "rally_t_end",
    "point_index",
    "attempt_role",
    "attempts_in_point",
)


class ScopeError(ValueError):
    """The declared execution scope and the root it is read against disagree."""


def validate_requested(requested: int | None) -> int | None:
    """A scope is opt-in and positive; everything else is a mistake, not a default."""
    if requested is None:
        return None
    if not isinstance(requested, int) or isinstance(requested, bool):
        raise ScopeError("--evaluation-first-attempts must be an integer")
    if requested < 1:
        raise ScopeError("--evaluation-first-attempts must be at least one")
    return requested


def ledger_rows(ledger: Path) -> list[dict[str, str]]:
    with Path(ledger).open(newline="") as handle:
        return list(csv.DictReader(handle))


def attempt_ids(rows: list[dict[str, str]]) -> list[int]:
    return [int(row["pt"]) for row in rows]


def _start_seconds(row: dict[str, str]) -> float | None:
    value = row.get("rally_t_start")
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate_ledger_order(rows: list[dict[str, str]]) -> None:
    """Refuse a ledger whose prefix is ambiguous.

    A duplicated identity or a row that starts before its predecessor would make "the first
    N rows" and "the first N attempts of the match" two different cohorts, and a later run
    could silently pick the other one.  Both are ledger faults, so the scope refuses rather
    than reordering a cohort the ledger already defined.
    """
    ids = attempt_ids(rows)
    if len(set(ids)) != len(ids):
        raise ScopeError("automatic ledger repeats an attempt identity; prefix is ambiguous")
    starts = [_start_seconds(row) for row in rows]
    known = [(index, value) for index, value in enumerate(starts) if value is not None]
    for (_, earlier), (_, later) in zip(known, known[1:], strict=False):
        if later < earlier:
            raise ScopeError(
                "automatic ledger rows are not in chronological order; "
                "a first-attempts prefix would not be the first attempts"
            )


def select_prefix(rows: list[dict[str, str]], requested: int) -> list[dict[str, str]]:
    """The first ``min(requested, L)`` ledger rows, unchanged and in ledger order."""
    validate_requested(requested)
    validate_ledger_order(rows)
    if not rows:
        raise ScopeError("automatic point ledger holds no attempts to scope")
    return rows[:requested]


def continuation_outside_scope(rows: list[dict[str, str]], selected: int) -> list[dict[str, Any]]:
    """Selected attempts whose score point continues into rows this scope does not run.

    Disclosure only: the physical point is incomplete for these attempts, so no complete
    physical-point claim may be made for them.  No numeric gate uses this.
    """
    excluded = rows[selected:]
    outside: dict[str, list[int]] = {}
    for row in excluded:
        index = row.get("point_index")
        if index in (None, ""):
            continue
        outside.setdefault(str(index), []).append(int(row["pt"]))
    disclosed = []
    for row in rows[:selected]:
        index = row.get("point_index")
        if index in (None, "") or str(index) not in outside:
            continue
        disclosed.append(
            {
                "pt": int(row["pt"]),
                "point_index": str(index),
                "attempt_role": row.get("attempt_role"),
                "remaining_attempts_outside_scope": sorted(outside[str(index)]),
            }
        )
    return disclosed


def build_scope(
    *,
    ledger: Path,
    rows: list[dict[str, str]],
    requested: int,
    match_id: str,
    source_video: Path,
    upstream_root: Path,
) -> dict[str, Any]:
    selected = select_prefix(rows, requested)
    count = len(selected)
    return {
        "schema": SCHEMA,
        "match_id": match_id,
        "observation_origin": "automatic",
        "labels_loaded": False,
        "fitted_states_as_inputs": False,
        "selection_rule": SELECTION_RULE,
        "requested_attempts": requested,
        "selected_attempts": count,
        "full_ledger_attempts": len(rows),
        "point_ledger": provenance.file_record(ledger, role="automatic_point_ledger"),
        "source_video": provenance.file_record(source_video, role="source_broadcast"),
        "upstream_root": provenance.portable_path(upstream_root),
        "implementation": provenance.file_record(Path(__file__), role="scope_implementation"),
        "selected_point_ids": [int(row["pt"]) for row in selected],
        "excluded_point_ids": [int(row["pt"]) for row in rows[count:]],
        "selected_windows": [
            {field: row.get(field) for field in WINDOW_FIELDS if field in row} for row in selected
        ],
        "continuation_outside_scope": continuation_outside_scope(rows, count),
    }


def encode(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def write_scope(path: Path, document: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encode(document))
    return path


def scope_path(root: Path) -> Path:
    return Path(root) / SCOPE_NAME


def load_scope(root: Path) -> dict[str, Any] | None:
    """The declared scope at ``root``, or ``None`` for an ordinary full-broadcast root."""
    path = scope_path(root)
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text())
    except ValueError as error:
        raise ScopeError(f"declared execution scope is unreadable: {error}") from error
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ScopeError("declared execution scope has an unknown schema")
    return document


def verify_scope(
    *,
    root: Path,
    match_id: str,
    ledger: Path,
    rows: list[dict[str, str]] | None = None,
    source_video: Path | None = None,
) -> dict[str, Any] | None:
    """Load the scope at ``root`` and check it still describes this ledger.

    Returns ``None`` when the root declares no scope.  Raises when the scope is present but
    its bound ledger has changed, its identities are not a prefix of the ledger, or it names
    another match: a consumer that cannot verify the scope must not fall back to the full
    roster, because the root it is reading was never materialized for one.
    """
    document = load_scope(root)
    if document is None:
        return None
    if str(document.get("match_id")) != match_id:
        raise ScopeError("declared execution scope names another match")
    record = provenance.file_record(ledger, role="automatic_point_ledger")
    if record.get("sha256") != document.get("point_ledger", {}).get("sha256"):
        raise ScopeError("declared execution scope does not bind this automatic point ledger")
    actual_rows = ledger_rows(ledger)
    if rows is not None and rows != actual_rows:
        raise ScopeError("consumer rows differ from the bound automatic ledger")
    rows = actual_rows
    requested = validate_requested(document.get("requested_attempts"))
    if requested is None:
        raise ScopeError("declared execution scope lacks its requested prefix")
    expected = select_prefix(rows, requested)
    ids = attempt_ids(rows)
    selected = document.get("selected_point_ids")
    count = len(expected)
    if selected != ids[:count] or document.get("excluded_point_ids") != ids[count:]:
        raise ScopeError("declared execution scope is not the ledger's first-attempts prefix")
    expected_fields = {
        "selected_attempts": count,
        "full_ledger_attempts": len(rows),
        "selected_windows": [
            {field: row.get(field) for field in WINDOW_FIELDS if field in row} for row in expected
        ],
        "continuation_outside_scope": continuation_outside_scope(rows, count),
        "selection_rule": SELECTION_RULE,
        "observation_origin": "automatic",
        "labels_loaded": False,
        "fitted_states_as_inputs": False,
    }
    if any(document.get(key) != value for key, value in expected_fields.items()):
        raise ScopeError("declared execution scope metadata differs from its bound ledger prefix")
    if source_video is not None and provenance.file_record(source_video).get("sha256") != (
        document.get("source_video", {}).get("sha256")
    ):
        raise ScopeError("declared execution scope names different source video bytes")
    from cv.pipeline.canonical_runner import POINT_MAP_NAME

    if ledger_rows(Path(root) / match_id / POINT_MAP_NAME) != expected:
        raise ScopeError("materialized point map differs from the declared ledger prefix")
    for name in (WORK_MANIFEST_NAME, CONVENTIONAL_MANIFEST_NAME):
        work = json.loads((Path(root) / name).read_text())
        matches = work.get("matches", [])
        if (
            len(matches) != 1
            or matches[0].get("id") != match_id
            or matches[0].get("point_ids") != selected
            or work.get("points_per_match") != count
        ):
            raise ScopeError("materialized work manifest differs from the declared ledger prefix")
    outside = set(point_directory_ids(Path(root) / match_id)) - set(selected)
    if outside:
        raise ScopeError("native attempt directories lie outside the declared execution scope")
    return document


def selected_limit(scope: dict[str, Any] | None, maximum: int | None) -> int | None:
    """The effective attempt limit for a consumer of a scoped root.

    An omitted maximum adopts the scope's own selection; a supplied maximum must agree with
    it, so a run never carries two conflicting selection rules.  A scope can never request
    a row the root did not materialize.
    """
    if scope is None:
        return maximum
    count = int(scope["selected_attempts"])
    if maximum is not None and maximum not in {count, scope.get("requested_attempts", count)}:
        raise ScopeError(
            f"--max-points {maximum} contradicts the declared execution scope of {count} "
            "selected attempts; omit it or pass the same value"
        )
    return count


def scope_stage_identity(scope: dict[str, Any]) -> list[str]:
    """Command fragment that separates a scoped materialization from a full one."""
    return [
        "--evaluation-first-attempts",
        str(scope["requested_attempts"]),
        "--evaluation-scope-ledger",
        str(scope["point_ledger"]["sha256"]),
        "--evaluation-selected",
        ",".join(str(value) for value in scope["selected_point_ids"]),
    ]


def point_directory_ids(match_directory: Path) -> list[int]:
    """Original attempt identities of the native frame directories already materialized."""
    frames = Path(match_directory) / "audit_frames_native_1080"
    if not frames.is_dir():
        return []
    ids = []
    for path in sorted(frames.iterdir()):
        name = path.name
        if path.is_dir() and name.startswith("pt"):
            if not name[2:].isdigit():
                raise ScopeError(f"malformed native attempt directory {path}; nothing is removed")
            ids.append(int(name[2:]))
    return ids


def reject_incompatible_root(
    *,
    root: Path,
    match_directory: Path,
    requested: int | None,
    selected_ids: list[int],
    expected_scope: dict[str, Any] | None = None,
) -> None:
    """Refuse a resume that would change what an existing output root means.

    A same-scope resume is ordinary.  A changed N, or a switch between full and scoped
    execution, would leave the root holding a mixture of two cohorts whose manifests and
    receipts disagree, so it fails with a new-root instruction.  Nothing is ever removed:
    the existing data is the user's.
    """
    existing = load_scope(root)
    if existing is None and requested is None:
        return
    materialized = point_directory_ids(match_directory)
    if existing is None:
        if requested is None:
            return
        from cv.pipeline.canonical_runner import POINT_MAP_NAME

        materialization_exists = any(
            path.exists()
            for path in (
                Path(root) / "broadcast_manifest.json",
                Path(root) / WORK_MANIFEST_NAME,
                Path(root) / CONVENTIONAL_MANIFEST_NAME,
                Path(match_directory) / POINT_MAP_NAME,
            )
        )
        if materialized or materialization_exists:
            raise ScopeError(
                f"{root} already holds a full-broadcast materialization or attempt directories; "
                "run the scope in a new output root -- nothing is removed here"
            )
        return
    if requested is None:
        raise ScopeError(
            f"{root} was materialized under a declared execution scope of "
            f"{existing['selected_attempts']} attempts; a full run needs a new output root"
        )
    if int(existing["requested_attempts"]) != int(requested) or [
        int(value) for value in existing["selected_point_ids"]
    ] != list(selected_ids):
        raise ScopeError(
            f"{root} was materialized for a different execution scope "
            f"(requested {existing['requested_attempts']}, now {requested}); "
            "run the new scope in a new output root -- nothing is removed here"
        )
    if expected_scope is not None and any(
        existing.get(key) != expected_scope.get(key)
        for key in (
            "point_ledger",
            "source_video",
            "selected_windows",
            "continuation_outside_scope",
        )
    ):
        raise ScopeError("bound scope inputs changed; use a new output root")
    outside = [value for value in materialized if value not in set(selected_ids)]
    if outside:
        raise ScopeError(
            f"{root} holds attempt directories outside its declared scope "
            f"(first {outside[:3]}); run in a new output root -- nothing is removed here"
        )


def preflight_root(root: Path, match_id: str, requested: int | None) -> None:
    """Refuse incompatible scope before the runner writes or reuses upstream stages."""
    existing = load_scope(root)
    if existing is None and (Path(root) / WORK_MANIFEST_NAME).exists():
        raise ScopeError("work manifest has no execution scope; use a new output root")
    selected = [] if existing is None else list(existing["selected_point_ids"])
    reject_incompatible_root(
        root=root,
        match_directory=Path(root) / match_id,
        requested=requested,
        selected_ids=selected,
    )


def guard_consumer_root(
    *,
    upstream_root: Path,
    match_id: str,
    ledger: Path,
    rows: list[dict[str, str]],
    maximum: int | None,
    source_video: Path | None = None,
) -> tuple[dict[str, Any] | None, int]:
    """Verify a scoped upstream root and return its scope and effective selection.

    Both full-identity consumers use this: they keep their strict full-ledger/full-manifest
    identity check and their complete roster, and only the number of rows they actually try
    is limited by the scope.
    """
    scope = verify_scope(
        root=upstream_root, match_id=match_id, ledger=ledger, rows=rows, source_video=source_video
    )
    limit = selected_limit(scope, maximum)
    total = len(rows)
    if scope is not None and int(scope["selected_attempts"]) > total:
        raise ScopeError("declared execution scope requests more attempts than the ledger holds")
    return scope, total if limit is None else min(limit, total)


def describe(scope: dict[str, Any] | None) -> dict[str, Any] | None:
    """Compact disclosure of a bound scope for a consumer's own report."""
    if scope is None:
        return None
    return {
        "schema": scope["schema"],
        "selection_rule": scope["selection_rule"],
        "requested_attempts": scope["requested_attempts"],
        "selected_attempts": scope["selected_attempts"],
        "full_ledger_attempts": scope["full_ledger_attempts"],
        "selected_point_ids": list(scope["selected_point_ids"]),
        "point_ledger_sha256": scope["point_ledger"]["sha256"],
        "continuation_outside_scope": scope.get("continuation_outside_scope", []),
        "note": (
            "scoped automatic pipeline measurement; cross-point court, camera, rig and "
            "ownership priors are fitted over the declared prefix only and are not the "
            "full-broadcast fits"
        ),
    }


def work_manifest(manifest: dict[str, Any], selected_ids: list[int]) -> dict[str, Any]:
    """The selected work roster in the same manifest schema as the full broadcast one."""
    matches = [
        {
            **match,
            "point_ids": [value for value in match["point_ids"] if value in set(selected_ids)],
        }
        for match in manifest["matches"]
    ]
    return {
        **manifest,
        "points_per_match": len(selected_ids),
        "matches": matches,
        "processing_scope": {
            "schema": SCHEMA,
            "selected_attempts": len(selected_ids),
            "full_ledger_attempts": manifest["points_per_match"],
        },
    }


def materialized_outputs(root: Path, match_directory: Path) -> list[Path]:
    """Every file a scoped materialization writes, for the stage receipt's output list."""
    from cv.pipeline.canonical_runner import POINT_MAP_NAME

    return [
        Path(match_directory) / POINT_MAP_NAME,
        Path(root) / "broadcast_manifest.json",
        Path(root) / WORK_MANIFEST_NAME,
        Path(root) / CONVENTIONAL_MANIFEST_NAME,
        scope_path(root),
    ]


def scope_summary_line(scope: dict[str, Any]) -> str:
    return (
        f"declared execution scope: {scope['selected_attempts']} of "
        f"{scope['full_ledger_attempts']} automatic attempts "
        f"({os.fspath(Path(scope['upstream_root']['path']))})"
    )
