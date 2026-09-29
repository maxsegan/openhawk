"""Stage actual automatic shared-S6 outputs and native video for held attempts.

Uses the backend's bound summary and original producer evidence. Never fits,
chooses a replacement trajectory, or certifies gates as visual correctness.
"""

from __future__ import annotations

import csv
from pathlib import Path

from cv.pipeline import provenance
from cv.viz import export_local_s6 as shared
from cv.viz import export_connected_3d as legacy
from scripts.shared_data import repository_root, resolve_shared_root

SCHEMA = "automatic_broadcast_shared_s6_v1"
GROUP = shared.AUTOMATIC_GROUP


def _identity(binding: dict) -> tuple:
    return tuple(binding.get(k) for k in ("path_base", "path", "sha256"))


def checked_binding(binding: dict, root: Path, repo: Path) -> Path:
    if not isinstance(binding, dict) or not binding.get("sha256"):
        raise ValueError("automatic export requires explicit artifact digests")
    return legacy._verify_binding(binding, root, repo)


def checked_source_video(summary: dict, root: Path, repo: Path) -> Path:
    binding = summary["source_video"]
    if binding.get("path_base") != "unconfigured_external":
        return checked_binding(binding, root, repo)
    from cv.pipeline.s6_automatic_observations import _resolve

    locator = summary.get("processing_video_path")
    if not isinstance(locator, str) or not Path(locator).is_absolute():
        raise ValueError("external source video requires its explicit processing_video_path")
    # Existing ancestry resolver compares basename, path base and full source SHA.
    return _resolve(binding, Path(locator))


def checked_summary(path: Path, root: Path, repo: Path) -> dict:
    document = legacy._read_json(path)
    if document.get("schema") != SCHEMA or document.get("observation_origin") != "automatic":
        raise ValueError("automatic broadcast shared-S6 summary required")
    if (
        document.get("runtime_frontier_model_calls") != 0
        or document.get("fitted_states_as_inputs") is not False
    ):
        raise ValueError("automatic export requires local inference without fitted inputs")

    # Neutral observation rows retain legacy slot names; their producer-bound
    # contents are checked by the stage validator, not a forbidden-key heuristic.
    def ancestry_row(row):
        cleaned = {k: v for k, v in row.items() if k != "observation_row"}
        if "children" in cleaned:
            cleaned["children"] = [ancestry_row(child) for child in cleaned["children"]]
        return cleaned

    ancestry_view = {**document, "attempts": [ancestry_row(row) for row in document["attempts"]]}
    provenance.assert_automatic_document(ancestry_view, context="automatic export summary")
    checked_source_video(document, root, repo)
    for name in ("manifest", "point_ledger", "policy_input"):
        checked_binding(document[name], root, repo)
    from cv.pipeline.s6_broadcast_backend import load_policy
    from cv.pipeline.s6_labeled_stage import shared_settings

    # Older summaries predate optional fields such as the quadratic reach-loss
    # default. Normalize the recorded policy by the same allowed defaults; an
    # explicit change or unknown field still differs from the bound input.
    normalized_recorded = {**document["policy"], **shared_settings(document["policy"])}
    if load_policy(checked_binding(document["policy_input"], root, repo)) != normalized_recorded:
        raise ValueError("automatic export policy differs from its bound input")
    with legacy._binding_path(document["point_ledger"], root, repo).open(newline="") as handle:
        ledger = list(csv.DictReader(handle))
    attempts = document["attempts"]
    if [r["ledger_row"] for r in attempts] != ledger:
        raise ValueError("automatic export must retain the original full ledger denominator")
    keys = []
    for row in attempts:
        expected_clip = f"pt{int(row['ledger_row']['pt']):04d}"
        expected_key = f"{document['match_id']}__{expected_clip}"
        if (
            row["clip"] != expected_clip
            or row["key"] != expected_key
            or Path(expected_key).name != expected_key
        ):
            raise ValueError("automatic attempt identity differs from original ledger")
        keys.append(row["key"])
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate automatic attempt identity")
    matches = legacy._read_json(legacy._binding_path(document["manifest"], root, repo))["matches"]
    matches = [r for r in matches if r["id"] == document["match_id"]]
    if len(matches) != 1 or set(matches[0]["point_ids"]) != {int(r["pt"]) for r in ledger}:
        raise ValueError("automatic manifest and ledger disagree")
    return document


def checked_preparation(attempt: dict, summary: dict, root: Path, repo: Path) -> tuple[dict, dict]:
    path = checked_binding(attempt["preparation"], root, repo)
    prepared = legacy._read_json(path)
    if prepared.get("schema") != "s6_automatic_preparation_v1":
        raise ValueError("automatic preparation receipt required")
    if prepared.get("row") != attempt.get("observation_row") or prepared.get(
        "evidence"
    ) != attempt.get("evidence"):
        raise ValueError("automatic preparation differs from summary")
    if prepared.get("runtime_model_calls") != 0 or prepared.get("fitted_states_read") is not False:
        raise ValueError("automatic preparation must not consume fitted states or models")
    producer_path = checked_binding(prepared["provenance"], root, repo)
    producer = provenance.load_provenance(producer_path, require_automatic=True)
    if _identity(summary["source_video"]) not in {_identity(r) for r in producer["source_videos"]}:
        raise ValueError("automatic producer does not bind this processing video")
    bound = {_identity(r) for r in producer["reused_artifacts"]}
    for binding in prepared["input_bindings"]:
        checked_binding(binding, root, repo)
        if _identity(binding) not in bound:
            raise ValueError("automatic preparation input lacks producer binding")
    for name in ("manifest", "point_ledger"):
        if _identity(summary[name]) not in bound:
            raise ValueError("automatic producer differs from broadcast source inputs")
    evidence = legacy._read_json(checked_binding(prepared["evidence"], root, repo))
    if (
        evidence.get("schema") != "s6_automatic_observed_evidence_v1"
        or evidence.get("automatic_ledger_row") != attempt["ledger_row"]
        or evidence.get("clip") != attempt["clip"]
        or evidence.get("match_id") != summary["match_id"]
    ):
        raise ValueError("automatic native evidence differs from its ledger attempt")
    # Linker verifies every picture digest; timebase helper checks all native epochs.
    legacy._native_timebase(evidence)
    if prepared["row"] is not None:
        if prepared["row"].get("key") != attempt["key"]:
            raise ValueError("automatic observation key differs from backend attempt")
        shared.checked_export_origin(prepared["row"])
        if prepared["row"].get("automatic_provenance") != prepared["provenance"]:
            raise ValueError("automatic observation producer differs from preparation")
    return prepared, evidence


def failed_native_evidence(attempt: dict, summary: dict, root: Path, repo: Path) -> dict | None:
    """Recover only original pictures after adapter failure, using actual bound PTS."""
    if (
        attempt.get("status") == "not_run_cap"
        or not attempt.get("extraction_receipt")
        or not summary.get("source_pts")
    ):
        return None
    from types import SimpleNamespace
    from cv.pipeline.s6_automatic_observations import _native_inventory

    receipt = checked_binding(attempt["extraction_receipt"], root, repo)
    pts = checked_binding(summary["source_pts"], root, repo)
    pts_receipt = legacy._read_json(checked_binding(summary["source_pts_receipt"], root, repo))
    if _identity(pts_receipt["output"]) != _identity(summary["source_pts"]) or _identity(
        pts_receipt["identity"]["source"]
    ) != _identity(summary["source_video"]):
        raise ValueError("native video PTS receipt differs from source or timestamps")
    frames = legacy._binding_path(attempt["native_frames_directory"], root, repo)
    if receipt.resolve() != (frames / "extraction_receipt.json").resolve():
        raise ValueError("native extraction receipt differs from picture directory")
    matches = legacy._read_json(legacy._binding_path(summary["manifest"], root, repo))["matches"]
    fps = float(next(r for r in matches if r["id"] == summary["match_id"])["source_fps"])
    inputs = SimpleNamespace(
        extraction_receipt=receipt,
        source_video=checked_source_video(summary, root, repo),
        source_pts=pts,
        frames_directory=frames,
        clip=attempt["clip"],
    )
    images = _native_inventory(inputs, fps, attempt["ledger_row"])
    return dict(source_pack=dict(fps=fps, images=images), ball_observations=[])


def export_held(
    attempt: dict,
    summary: dict,
    summary_path: Path,
    out: Path,
    root: Path,
    repo: Path,
    label: str,
    prepared: dict | None = None,
    evidence: dict | None = None,
) -> dict:
    entry = dict(
        key=attempt["key"],
        point=attempt["key"],
        held=True,
        status=attempt["status"],
        reason=attempt.get("reason"),
        declared_flights=None,
        source_group=GROUP,
        gate_accepted=False,
    )
    if evidence is None:
        evidence = failed_native_evidence(attempt, summary, root, repo)
    if evidence is None:
        entry["frames_available"] = False
        entry["video_unavailable_reason"] = "No verified native evidence binding is available yet."
        return entry
    images = evidence["source_pack"]["images"]
    native = [int(r["frame"]) for r in images if r["clip"] == attempt["clip"]]
    if not native:
        raise ValueError("automatic held evidence has no original pictures")
    observation_document = {
        "match_id": summary["match_id"],
        "source_pack": evidence["source_pack"],
        "attempt": {
            "clip": attempt["clip"],
            "first_contact_frame": min(native),
            "ending_frame": max(native),
        },
        "ball": {
            "records": [{"clip": attempt["clip"], "frames": evidence.get("ball_observations", [])}]
        },
    }
    origin = dict(
        human_derived=False,
        automatic_inference_eligible=False,
        observation_origin="automatic",
        automatic_provenance=prepared["provenance"] if prepared else None,
        automatic_evidence=prepared["evidence"] if prepared else None,
        native_extraction=attempt.get("extraction_receipt"),
    )
    row = dict(
        key=attempt["key"],
        declared_flights=None,
        labels=prepared["evidence"] if prepared else attempt["extraction_receipt"],
    )
    result = dict(key=attempt["key"], status=attempt["status"], reason=attempt.get("reason"))
    entry = shared._export_observation_document(
        row,
        observation_document,
        result,
        entry,
        checked_binding(attempt["preparation"], root, repo) if prepared else summary_path,
        summary_path,
        out,
        root,
        label,
        origin,
    )
    doc_path = out / "data" / entry["file"]
    doc = legacy._read_json(doc_path)
    doc["review_context"]["kind"] = "automatic_observations_only"
    doc["review_context"]["physical_ending_known"] = False
    doc["provenance"].pop("result", None)
    doc["provenance"]["preparation"] = attempt.get("preparation")
    if not prepared:
        doc["provenance"]["source_pts"] = summary["source_pts"]
        doc["provenance"]["source_pts_receipt"] = summary["source_pts_receipt"]
        doc["provenance"]["input_bindings"] = {"native_extraction": attempt["extraction_receipt"]}
    doc["quality"]["footnote"] = (
        "Original automatic observations only. No fitted output; no complete-point or reference-flight claim."
    )
    legacy._write_json(doc_path, doc)
    return entry


def export_attempts(summary: dict, root: Path, repo: Path) -> list[dict]:
    """Expand validated child outputs without replacing the original parent ledger."""
    result = []
    for parent in summary["attempts"]:
        if "children" not in parent:
            result.append(parent)
            continue
        prepared = legacy._read_json(checked_binding(parent["preparation"], root, repo))
        plan = legacy._read_json(checked_binding(parent["evidence"], root, repo))
        children = parent["children"]
        if (
            prepared.get("schema") != "s6_automatic_service_parent_preparation_v1"
            or prepared.get("evidence") != parent["evidence"]
            or prepared.get("original_parent_retained") is not True
            or plan.get("schema") != "automatic_service_attempt_scope_v1"
            or not children
            or len(children) != len(prepared["children"])
            or len(children) != len(plan["children"])
        ):
            raise ValueError("automatic child outputs differ from source-bound parent plan")
        for index, (child, source, scope) in enumerate(
            zip(children, prepared["children"], plan["children"], strict=True), 1
        ):
            if (
                child.get("key") != f"{parent['key']}__a{index:02d}"
                or child.get("parent_key") != parent["key"]
                or child.get("child_index") != index
                or child.get("clip") != parent["clip"]
                or child.get("ledger_row") != parent["ledger_row"]
                or child.get("service_attempt_scope") != scope
                or legacy._read_json(checked_binding(child["preparation"], root, repo)) != source
            ):
                raise ValueError("automatic child identity or membership changed")
            result.append(child)
    return result


def export_run(
    run: Path,
    out: Path,
    cases: list[str] | None = None,
    merge: bool = False,
    run_label: str | None = None,
) -> dict:
    repo = repository_root()
    root = resolve_shared_root(repo, None)
    if out.resolve() == (root / "processed/review_queue_v1/portal/3d").resolve():
        raise ValueError("export to staging; coordinator reviews before deployment")
    path = run / "summary.json"
    summary = checked_summary(path, root, repo)
    attempts = export_attempts(summary, root, repo)
    requested = set(cases or [])
    if requested - {r["key"] for r in summary["attempts"]}:
        raise ValueError("unknown automatic case")
    if out.exists() and not merge:
        raise ValueError("fresh staging directory required unless --merge is explicit")
    prior = (
        legacy._read_json(out / "data/index.json")
        if merge and (out / "data/index.json").exists()
        else {}
    )
    shared.copy_static(out)
    entries, held = [], []
    label = (
        run_label
        or f"Automatic upstream + shared S6 · {summary.get('execution_scope', 'declared source')}"
    )
    for attempt in attempts:
        if requested and attempt.get("parent_key", attempt["key"]) not in requested:
            continue
        prepared = evidence = None
        if attempt.get("preparation"):
            prepared, evidence = checked_preparation(attempt, summary, root, repo)
        if attempt.get("result"):
            if not prepared or prepared["row"] is None:
                raise ValueError(
                    "automatic stage output requires a validated prepared observation row"
                )
            result = checked_binding(attempt["result"], root, repo)
            expected = run / "attempts" / attempt.get("parent_key", attempt["key"])
            if "parent_key" in attempt:
                expected = expected / f"child_{attempt['child_index']:02d}"
            if result.resolve() != (expected / "stage/result.json").resolve():
                raise ValueError("automatic result is not this attempt's stage output")
            entry = shared.export_case(prepared["row"], result, path, out, root, repo, label)
        else:
            entry = export_held(attempt, summary, path, out, root, repo, label, prepared, evidence)
        if "parent_key" in attempt:
            entry.update(
                parent_key=attempt["parent_key"],
                child_index=attempt["child_index"],
                service_attempt_scope=attempt["service_attempt_scope"],
            )
            if entry.get("file"):
                document_path = out / "data" / entry["file"]
                document = legacy._read_json(document_path)
                document.setdefault("review_context", {}).update(
                    automatic_parent_key=attempt["parent_key"],
                    service_attempt_scope=attempt["service_attempt_scope"],
                    video_context_is_not_trajectory=True,
                )
                legacy._write_json(document_path, document)
        (held if entry.get("held") or not entry.get("file") else entries).append(entry)
    selected_keys = {
        r["key"] for r in attempts if not requested or r.get("parent_key", r["key"]) in requested
    }
    old = [
        r
        for r in prior.get("points", [])
        if not (r.get("source_group") == GROUP and r.get("key") in selected_keys)
    ]
    prior_held = [
        r
        for r in prior.get("automatic_s6", {}).get("held_before_fitting", [])
        if r.get("key") not in selected_keys
    ]
    report = dict(
        run=shared.record(path),
        held_before_fitting=[*held, *prior_held],
        exported_attempts=len(entries),
        requested_attempts=len(entries) + len(held),
        original_automatic_attempt_denominator=len(summary["attempts"]),
        source_counts=summary.get("counts"),
        reference_flight_denominator=None,
        useful_complete_point_yield=None,
        scope=label,
        **(
            {
                "original_parents": [
                    {
                        "key": r["key"],
                        "status": r["status"],
                        "children": [c["key"] for c in r.get("children", [])],
                    }
                    for r in summary["attempts"]
                ]
            }
            if any(r.get("children") for r in summary["attempts"])
            else {}
        ),
    )
    index = {
        **prior,
        "schema": legacy.INDEX_SCHEMA,
        "collection": "canonical_3d_viewer",
        "source_groups": [
            {"id": GROUP, "label": label},
            *[g for g in prior.get("source_groups", []) if g["id"] != GROUP],
        ],
        "points": [*entries, *old],
        "count": len(entries) + len(old),
        "automatic_s6": report,
    }
    legacy._write_json(out / "data/index.json", index, pretty=True)
    legacy._write_json(out / "automatic_s6_export.json", report, pretty=True)
    return report
