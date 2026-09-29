"""Recover omitted producer-code ancestry from a completed, recorded clean run.

This writes a new, explicitly retrospective provenance document. It never edits
original stage receipts or guesses a compatible Git revision from file contents.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

from cv.pipeline import artifact_cache, paths, provenance

CONFIG_KEY = "historical_source_recovery"


def _identity(record: dict) -> tuple:
    return tuple(record.get(k) for k in ("path_base", "path", "sha256", "bytes"))


def verified_sources(document: dict, parent_record: dict, source_video: Path) -> dict:
    """Return source proof only for exact receipts bound by completed-run evidence."""
    from cv.pipeline.s6_automatic_observations import (
        _checked_record,
        historical_implementation_input,
    )

    recovery = document.get("configuration", {}).get(CONFIG_KEY)
    if recovery is None:
        return _recorded_run_sources(document, parent_record, source_video)
    if not isinstance(recovery, dict) or recovery.get("recovered_after_run") is not True:
        raise ValueError("producer source recovery must declare retrospective timing")
    try:
        stamp = datetime.fromisoformat(recovery["recovered_utc"])
        if stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
            raise ValueError("timezone missing")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("producer source recovery requires an explicit UTC timestamp") from error
    source = recovery.get("source", {})
    if not isinstance(source, dict) or source.get("dirty") is not False:
        raise ValueError("producer source recovery requires recorded clean source")
    bound = {_identity(r) for r in document["reused_artifacts"]}
    for field in ("completed_status", "clean_run_assertion"):
        record = recovery.get(field, {})
        if _identity(record) not in bound:
            raise ValueError("producer source recovery evidence is not bound")
        evidence = _checked_record(record, source_video)
        if field == "completed_status":
            status = json.loads(evidence.read_text())
    if (
        status.get("status") != "complete"
        or status.get("normal_runner_stage_executed") is not True
        or status.get("producer_bindings_verified") is not True
        or status.get("source_commit") != source.get("commit")
    ):
        raise ValueError("producer source recovery needs matching completed execution evidence")
    results = {}
    for binding in document["reused_artifacts"]:
        if binding.get("role") != "automatic_stage_receipt":
            continue
        receipt_path = _checked_record(binding, source_video)
        receipt = json.loads(receipt_path.read_text())
        identity = receipt.get("identity", {})
        recorded = status.get("stage_receipts", {}).get(identity.get("stage"), {})
        if _identity(recorded) != _identity(binding):
            raise ValueError("producer source recovery receipt differs from completed status")
        unsigned = {k: v for k, v in identity.items() if k != "fingerprint"}
        if receipt.get("schema") != artifact_cache.SCHEMA or artifact_cache._digest_json(
            unsigned
        ) != identity.get("fingerprint"):
            raise ValueError("producer source recovery receipt fingerprint changed")
        closure = identity.get("code", [])
        if not closure:
            raise ValueError("producer source recovery needs a declared code closure")
        # Verify every declared implementation file, not only the one whose
        # current checkout differs. Observations/models never take this route.
        historical = {
            **identity,
            "configuration": {"inference_source": source},
        }
        for code in closure:
            if (
                code.get("kind") != "file"
                or historical_implementation_input(code, historical) is None
            ):
                raise ValueError("producer source recovery has an unsupported code closure")
        key = (binding["sha256"], binding["bytes"])
        results[key] = dict(
            source=source,
            recovery_provenance=parent_record,
            completed_status=recovery["completed_status"],
            clean_run_assertion=recovery["clean_run_assertion"],
            recovered_utc=recovery["recovered_utc"],
            closure_files=len(closure),
        )
    if not results:
        raise ValueError("producer source recovery requires a bound executed stage receipt")
    return results


def _recorded_run_sources(document: dict, parent_record: dict, source_video: Path) -> dict:
    """Carry an existing explicit producer binding, never an enclosing Git guess."""
    from cv.pipeline.s6_automatic_observations import (
        _checked_record,
        historical_implementation_input,
    )

    config = document.get("configuration", {})
    if not (config.get("producer_git") and config.get("original_stage_run")):
        return {}
    source = config["producer_git"]
    run_record = config["original_stage_run"]
    if _identity(run_record) not in {_identity(r) for r in document["reused_artifacts"]}:
        raise ValueError("recorded producer run is not bound")
    run = json.loads(_checked_record(run_record, source_video).read_text())
    provenance.assert_automatic_document(run["provenance"], context="recorded producer run")
    if (
        source.get("dirty") is not False
        or run["provenance"].get("git") != source
        or run.get("status") != "ok"
        or not run.get("ended_at")
        or run.get("stage") != config.get("original_stage_run_name")
        or config.get("receipt_written_by_runner_stage") is not True
    ):
        raise ValueError("recorded producer run source/status differs")
    module = config.get("producer_module")
    results = {}
    for binding in document["reused_artifacts"]:
        if not str(binding.get("role", "")).endswith("inference_receipt"):
            continue
        receipt = json.loads(_checked_record(binding, source_video).read_text())
        identity = receipt.get("identity", {})
        command = identity.get("command", [])
        if not module or not any(
            command[i : i + 2] == ["-m", module] for i in range(len(command) - 1)
        ):
            raise ValueError("recorded producer receipt command differs")
        unsigned = {k: v for k, v in identity.items() if k != "fingerprint"}
        if receipt.get("schema") != artifact_cache.SCHEMA or artifact_cache._digest_json(
            unsigned
        ) != identity.get("fingerprint"):
            raise ValueError("recorded producer receipt fingerprint changed")
        output_records = {
            (r.get("path_base"), r.get("path")): r
            for r in run.get("outputs", {}).values()
            if isinstance(r, dict)
        }
        if not output_records or not receipt.get("outputs"):
            raise ValueError("recorded producer receipt lacks run outputs")
        for output in receipt["outputs"]:
            original = output_records.get((output.get("path_base"), output.get("path")))
            if original is None or any(
                key in original and original[key] != output.get(key) for key in ("sha256", "bytes")
            ):
                raise ValueError("recorded producer receipt outputs differ from run")
        proof = dict(
            source=source,
            verification="bound_original_stage_run_source",
            original_run_output_binding=(
                "full_identity"
                if all("sha256" in r and "bytes" in r for r in output_records.values())
                else "path_only"
            ),
            producer_provenance=parent_record,
            original_stage_run=run_record,
        )
        closure = identity.get("code", [])
        if not closure or any(
            historical_implementation_input(r, identity, proof) is None for r in closure
        ):
            raise ValueError("recorded producer run has unsupported code closure")
        proof["closure_files"] = len(closure)
        results[(binding["sha256"], binding["bytes"])] = proof
    return results


def recover_completed_producer(
    *,
    source_checkout: Path,
    completed_status: Path,
    clean_run_assertion: Path,
    stages: tuple[str, ...],
    source_video: Path,
    output: Path,
    parents: tuple[Path, ...] = (),
    receipt_roles: dict[str, str] | None = None,
) -> Path:
    """Write additive evidence for explicitly selected stages of the recorded run."""
    from cv.pipeline.s6_automatic_observations import _checked_record

    if output.exists():
        raise ValueError("producer source recovery requires a new output")
    source = provenance.git_record(source_checkout)
    status = json.loads(completed_status.read_text())
    status_record = provenance.file_record(completed_status, role="completed_producer_status")
    assertion_record = provenance.file_record(clean_run_assertion, role="clean_run_assertion")
    receipts, outputs = [], []
    for stage in stages:
        binding = status.get("stage_receipts", {}).get(stage)
        if not isinstance(binding, dict):
            raise ValueError("completed producer did not bind the requested stage")
        path = _checked_record(binding, source_video)
        receipts.append(provenance.file_record(path, role="automatic_stage_receipt"))
        if role := (receipt_roles or {}).get(stage):
            receipts.append(provenance.file_record(path, role=role))
        for record in json.loads(path.read_text())["outputs"]:
            if record.get("kind") == "file":
                artifact = _checked_record(record, source_video)
                outputs.append(provenance.file_record(artifact, role="automatic_producer_output"))
    document = provenance.build_provenance(
        root=paths.REPO_ROOT,
        mode=provenance.AUTOMATIC_MODE,
        source_videos=[provenance.file_record(source_video, role="processing_video")],
        configuration={
            CONFIG_KEY: dict(
                recovered_after_run=True,
                recovered_utc=datetime.now(timezone.utc).isoformat(),
                source=source,
                completed_status=status_record,
                clean_run_assertion=assertion_record,
            )
        },
        reused_artifacts=[
            status_record,
            assertion_record,
            *[provenance.file_record(p, role="automatic_parent_provenance") for p in parents],
            *receipts,
            *outputs,
        ],
    )
    verified_sources(document, {}, source_video)
    return provenance.write_provenance(output, document)
