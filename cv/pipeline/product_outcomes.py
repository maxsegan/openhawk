"""Per-match outcome records: what happened to every point, and why, in stable reason codes.

    python -m cv.pipeline.product_outcomes MATCH_DIR [--match-id KEY] [--out OUTCOMES.json]

``OUTCOMES.json`` (``product_outcomes_v1``) sits beside ``MATCH_RESULT.json`` and
``RECEIPT.json`` in a product-runner match directory. It is written by the runner at
``assemble`` (from the S6 results assemble already reads), refreshed after every later step,
and written on failure, so a match that never reaches ``assemble`` still says where it
stopped and why. It records:

- ``identity``: runner code commit, upstream stage fingerprints, model weight hashes, the
  effective S6 policy (switch values and sha256, ``PIPELINE_COMPONENT_POLICY`` included),
  event adjudication mode;
- ``source``: provider, fps, resolution, cadence decision;
- ``stages``: one row per stage (source admission, point segmentation, camera/court check,
  tracking, point validity, events, S6 fit, cascade, refit, post-point tag, portal, compact)
  with ``status``, a stable ``reason`` code and counts;
- ``attempts``: per ledger attempt (split children as their own rows) the upstream gate
  results, the S6 status and reason, each component's status and reason, each original
  flight slot's outcome and reason (``accepted``, ``gate:<check>``, ``plan:<reason>``, a
  component failure code), the post-point tag, and where its retained S6 fit inputs live
  (``reprocess``), so ``cv.pipeline.outcome_index`` can select and refit only those points.

Reason codes come from :func:`reason_code`; unknown messages fall back to
``execution_failed:<ExceptionType>`` or a slug, never to free text. The record is a view of
existing artifacts: nothing here decides an outcome. No evaluation label is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

from cv.pipeline import player_names, provenance, switch_fallbacks

SCHEMA = "product_outcomes_v1"
FILE = "OUTCOMES.json"
REPO = Path(__file__).resolve().parents[2]
STEPS = ("upstream", "fit", "cascade", "refit", "assemble", "portal", "compact")
S6_ARCHIVE = "archive/s6_stages.tar.zst"
UPSTREAM_ARCHIVE = "archive/upstream.tar.zst"
UNZSTD = "zstd -d --long=30"
WEIGHT_SUFFIXES = (".pt", ".pth", ".onnx", ".safetensors", ".ckpt", ".tar", ".bin", ".engine")
UPSTREAM_GATES = (
    "court_geometry_point_gate_v1.json",
    "point_validity_gate_v1.json",
    "untouched_tracking_point_gate_v1.json",
)

#: Message patterns to stable reason codes, first match wins. Codes are the vocabulary of
#: ``yield_report`` and ``reprocess_select``: extend this table, never rename a code.
REASON_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"source cadence rejected: (\w+)", r"cadence_refused:\1"),
    (r"source resolution rejected: (\w+)", r"resolution_refused:\1"),
    (r"component cameras differ", "component_cameras_differ"),
    (r"lacks fixed train/check coverage", "train_check_coverage_missing"),
    (r"no completed candidate after \d+ numerical failures", "seed_failure"),
    (r"search budget exhausted|search_exhausted|exhausted_split", "search_exhausted"),
    (r"component mode requires explicit search_seconds", "search_budget_invalid"),
    (r"no contact inside observation scope", "events_no_contact_in_scope"),
    (r"no contact before point_end", "events_no_contact_before_point_end"),
    (
        r"declared leading physical evidence requires an unresolved observation scope",
        "events_leading_evidence_conflict",
    ),
    (r"automatic_event_preparation_refused: (.+)", "events_refused"),
    (r"camera artifact has no rows in the evaluation window", "camera_no_rows_in_window"),
    (r"TimeoutExpired|timed out after", "execution_failed:timeout"),
    (r"CUDA out of memory|OutOfMemoryError", "execution_failed:gpu_oom"),
    (r"No space left on device", "execution_failed:disk_full"),
    (r"shared S6 output does not bind", "execution_failed:binding"),
)
_EXCEPTION = re.compile(
    r"(?:^|[\s:.(])((?:[A-Za-z_][\w]*\.)*([A-Z]\w*(?:Error|Exception|Failure|Expired)))"
)


def reason_code(text: str | None) -> str | None:
    """A stable reason code for a status message (``None`` stays ``None``)."""
    if not text:
        return None
    text = str(text)
    for pattern, code in REASON_PATTERNS:
        found = re.search(pattern, text)
        if found:
            return found.expand(code) if "\\" in code else code
    exception = _EXCEPTION.search(text)
    if exception:
        return f"execution_failed:{exception.group(2)}"
    words = re.findall(r"[a-z0-9]+", text.lower())[:6]
    return "_".join(words) or "unknown"


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _save(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=1, default=str) + "\n")
    temporary.replace(path)


def canonical_sha256(document) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


# --------------------------------------------------------------------------- attempts


def component_outcomes(result: dict) -> list[dict]:
    """Status and reason code of every S6 component of one attempt result."""
    rows = []
    for component in result.get("component_results") or []:
        inner = component.get("result") or {}
        status = inner.get("status")
        failed = status and "failed" in status
        rows.append(
            {
                "index": component.get("component_index"),
                "status": status,
                "reason": reason_code(inner.get("reason")) if failed else None,
                "accepted_flights": (inner.get("verdict") or {}).get("accepted_flight_count"),
                "wall_seconds": inner.get("wall_seconds"),
            }
        )
    return rows


def flight_outcomes(result: dict, components: list[dict] | None = None) -> list[dict]:
    """One row per original flight slot of the S6 verdict with its reason code.

    ``accepted``; else the first failed physics check as ``gate:<check>``; else a slot the
    plan could not cover as ``plan:<reason>``; a slot whose component did not emit takes the
    component's failure code (``component_cameras_differ``, ``seed_failure`` ...).
    """
    failed = {c["index"]: c["reason"] for c in components or [] if c.get("reason")}
    rows = []
    for flight in (result.get("verdict") or {}).get("flights") or []:
        gates = list(flight.get("failures") or [])
        plan = list(flight.get("reasons") or [])
        if flight.get("accepted"):
            reason = "accepted"
        elif gates:
            reason = f"gate:{gates[0]}"
        elif plan == ["component_execution_did_not_emit"]:
            reason = failed.get(flight.get("component_index")) or "component_did_not_emit"
        elif plan:
            reason = f"plan:{plan[0]}"
        else:
            reason = "rejected_unspecified"
        rows.append(
            {
                "index": flight.get("original_flight_index", flight.get("flight_index")),
                "component": flight.get("component_index"),
                "role": flight.get("role"),
                "start_frame": flight.get("start_frame"),
                "end_frame": flight.get("end_frame"),
                "accepted": bool(flight.get("accepted")),
                "reason": reason,
                "gates": gates,
                "plan_reasons": plan,
            }
        )
    return rows


def primary_reason(fit_status: str | None, held: str | None, flights: list[dict]) -> str:
    """The one code an attempt is counted under in a yield breakdown."""
    if fit_status and fit_status != "completed":
        return reason_code(held) or fit_status
    if any(f["accepted"] for f in flights):
        return "accepted"
    if not flights:
        return "no_flight_slots"
    counts = Counter(f["reason"] for f in flights)
    return max(counts, key=lambda code: (counts[code], code.startswith("component")))


def attempt_outcome(
    attempt: dict,
    result: dict | None,
    *,
    final_source: str | None,
    refit_status: str | None,
    final_flights: list[dict] | None,
    point_end: dict | None,
    gates: dict | None = None,
    fit_job: dict | None = None,
) -> dict:
    """One attempt's record. ``final_flights`` are the assembled (tagged) accepted flights."""
    components = component_outcomes(result) if result else []
    flights = flight_outcomes(result, components) if result else []
    tagged = {
        (round(float(f["start_frame"] or 0), 1)): f.get("after_point_end")
        for f in final_flights or []
    }
    for flight in flights:
        if flight["accepted"]:
            flight["after_point_end"] = tagged.get(round(float(flight["start_frame"] or 0), 1))
    verdict = (result or {}).get("verdict") or {}
    ledger = attempt.get("ledger_row") or attempt.get("ledger") or {}
    status = attempt.get("status") or attempt.get("first_fit_status")
    held = attempt.get("reason") if "reason" in attempt else attempt.get("held_reason")
    final_status = (result or {}).get("status")
    if final_source is None:
        fit_stage = "s6_preparation" if status and status.startswith("preparation") else "s6_fit"
    else:
        fit_stage = "s6_fit"
    return {
        "key": attempt["key"],
        "parent_key": attempt.get("parent_key", attempt["key"]),
        "clip": attempt.get("clip"),
        "point_index": ledger.get("point_index"),
        "attempt_role": ledger.get("attempt_role"),
        "ledger_confidence": ledger.get("confidence"),
        "score_source": ledger.get("score_source"),
        "upstream_gates": gates or {},
        "first_fit": {"status": status, "reason": reason_code(held), "message": held},
        "final_source": final_source,
        "refit_status": refit_status,
        "result_status": final_status,
        "stage": fit_stage,
        "reason": primary_reason(status if final_source is None else "completed", held, flights),
        "slots": verdict.get("original_slot_count") or len(flights),
        "accepted_flights": len(final_flights or []),
        "accepted_flights_after_point_end": sum(
            bool(f.get("after_point_end")) for f in final_flights or []
        ),
        "complete_point": bool(verdict.get("complete_point")),
        "components": components,
        "flights": flights,
        "point_end": (
            {"evidence": point_end.get("evidence"), "frame": point_end.get("frame")}
            if point_end
            else None
        ),
        "reprocess": fit_job or {},
    }


# --------------------------------------------------------------------------- upstream


def upstream_gates(upstream: Path) -> dict[str, dict]:
    """Per clip: court geometry (camera/paint), point validity and tracking gate results."""
    gates: dict[str, dict] = {}
    documents = {name: upstream / name for name in UPSTREAM_GATES}
    if documents["court_geometry_point_gate_v1.json"].is_file():
        for row in _load(documents["court_geometry_point_gate_v1.json"]).get("rows") or []:
            gates.setdefault(row["clip"], {}).update(
                court_valid=row.get("valid"), court_reasons=row.get("reasons") or []
            )
    if documents["point_validity_gate_v1.json"].is_file():
        for row in _load(documents["point_validity_gate_v1.json"]).get("rows") or []:
            gates.setdefault(row["clip"], {}).update(
                validity=row.get("decision"),
                validity_reasons=row.get("reasons") or [],
                hard_camera_failure=row.get("hard_camera_failure"),
                hard_timing_failure=row.get("hard_timing_failure"),
                cadence=row.get("frame_cadence_decision"),
            )
    if documents["untouched_tracking_point_gate_v1.json"].is_file():
        for row in _load(documents["untouched_tracking_point_gate_v1.json"]).get("rows") or []:
            gates.setdefault(row["clip"], {}).update(
                tracking=row.get("arc_decision_summary"),
                tracking_retained_frame_rate=row.get("arc_retained_frame_rate"),
            )
    return gates


def upstream_identity(match_upstream: Path) -> dict:
    """Model weight hashes and per-stage fingerprints from the upstream stage receipts."""
    models: dict[str, str] = {}
    fingerprints: dict[str, str] = {}

    def walk(node):
        if isinstance(node, dict):
            path = node.get("path")
            if (
                isinstance(path, str)
                and node.get("sha256")
                and path.endswith(WEIGHT_SUFFIXES)
                and "/fit/" not in path
            ):
                models[path] = node["sha256"]
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for receipt in sorted((match_upstream / "run_manifests" / "stage_receipts").glob("*.json")):
        identity = _load(receipt).get("identity") or {}
        fingerprint = identity.get("fingerprint")
        if fingerprint:
            fingerprints[receipt.stem] = fingerprint
        walk(identity)
    return {"models": dict(sorted(models.items())), "upstream_stage_fingerprints": fingerprints}


def source_record(match_upstream: Path, video: str | None) -> dict:
    record: dict = {"video": video, "provider": provider_of(video)}
    path = match_upstream / "source_integrity.json"
    if not path.is_file():
        return record
    integrity = _load(path)
    cadence = integrity.get("cadence") or {}
    record.update(
        fps=integrity.get("fps"),
        nominal_fps=integrity.get("nominal_fps"),
        width=integrity.get("width"),
        height=integrity.get("height"),
        duration_seconds=integrity.get("duration_seconds"),
        resolution_valid=integrity.get("resolution_valid"),
        cadence_decision=cadence.get("decision"),
        cadence_reason=cadence.get("reason") or None,
        processing_fps=cadence.get("processing_fps"),
    )
    return record


def provider_of(video: str | None) -> str | None:
    """Prefix of the video file name before its last underscore (``prefix_1234.mp4``)."""
    if not video:
        return None
    stem = Path(video).stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


# --------------------------------------------------------------------------- policy


def policy_identity(policy: dict | None) -> dict:
    """Effective S6 policy: switch values and sha256, plus the in-code component policy."""
    from cv.pipeline.s6_labeled_stage import PIPELINE_COMPONENT_POLICY

    record = {
        "pipeline_component_policy_sha256": canonical_sha256(PIPELINE_COMPONENT_POLICY),
        "pipeline_component_policy_note": "hash of the in-code policy of the builder's commit",
    }
    if policy:
        record["s6_policy_sha256"] = canonical_sha256(policy)
        record["s6_policy"] = policy
        record["component_switches"] = {
            key: policy.get(key) for key in sorted(PIPELINE_COMPONENT_POLICY) if key in policy
        }
    return record


# --------------------------------------------------------------------------- stages


def _marker(out: Path, step: str) -> dict | None:
    path = out / "steps" / f"{step}.json"
    try:
        return _load(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _scalars(detail: dict | None) -> dict:
    """Step detail without bulky lists (their lengths stand in)."""
    return {
        key: len(value) if isinstance(value, list) else value
        for key, value in (detail or {}).items()
        if not isinstance(value, dict) or key == "decisions"
    }


def step_rows(out: Path) -> dict[str, dict]:
    rows = {}
    for step in STEPS:
        marker = _marker(out, step)
        if marker is None:
            rows[step] = {"status": "not_run"}
            continue
        rows[step] = {
            "status": marker.get("status"),
            "reason": reason_code(marker.get("error"))
            if marker.get("status") == "failed"
            else None,
            "message": (marker.get("error") or "")[-400:] or None,
            "wall_seconds": marker.get("wall_seconds"),
            "gpu_seconds": marker.get("gpu_seconds"),
            "detail": _scalars(marker.get("detail"))
            if step in ("portal", "compact", "cascade")
            else None,
            # Switches the step could not honour on its box (cv.pipeline.switch_fallbacks).
            "fallbacks": marker.get("fallbacks"),
        }
    return rows


def step_switches(out: Path) -> dict[str, dict | None]:
    """Per step marker (beam and harvest too): the switch values, runtime pin and box
    environment it ran under (``switch_fallbacks.snapshot``; None before 2026-10-02).
    ``product_reprocess rerun`` replays them."""
    found = {}
    for path in sorted((out / "steps").glob("*.json")):
        marker = _marker(out, path.stem)
        if marker is not None:
            found[path.stem] = marker.get("switches")
    return found


def stage_rows(out: Path, match_id: str, source: dict, attempts: list[dict], steps: dict) -> dict:
    """Stage-level status, reason and counts, product order."""
    match_upstream = out / "upstream" / match_id
    stages: dict[str, dict] = {}
    upstream_failed = steps["upstream"]["status"] == "failed"
    reason = steps["upstream"].get("reason") if upstream_failed else None
    if source.get("cadence_decision") == "rejected":
        admission = {"status": "refused", "reason": f"cadence_refused:{source['cadence_reason']}"}
    elif source.get("resolution_valid") is False:
        admission = {"status": "refused", "reason": "resolution_refused"}
    elif source.get("cadence_decision"):
        admission = {"status": "ok", "reason": None}
    elif upstream_failed and reason and reason.startswith(("cadence", "resolution")):
        admission = {"status": "refused", "reason": reason}
    else:
        admission = {"status": "not_run", "reason": None}
    stages["source_admission"] = {**admission, "counts": {"fps": source.get("fps")}}
    ledger = match_upstream / "automatic_point_ledger_v1.json"
    if ledger.is_file():
        document = _load(ledger)
        stages["point_segmentation"] = {
            "status": "ok" if document.get("attempts") else "empty",
            "reason": None if document.get("attempts") else "ledger_empty",
            "counts": {
                key: document.get(key)
                for key in (
                    "attempts",
                    "score_points",
                    "score_abstentions",
                    "high_confidence",
                    "provisional",
                    "multi_attempt_points",
                )
            },
        }
    else:
        stages["point_segmentation"] = {
            "status": "failed" if upstream_failed else "not_run",
            "reason": reason if upstream_failed else None,
            "counts": {},
        }

    def gate_counts(field: str, reasons_field: str | None = None) -> dict:
        values = Counter(str(a["upstream_gates"].get(field)) for a in attempts)
        reasons = Counter(
            r for a in attempts for r in (a["upstream_gates"].get(reasons_field) or [])
        )
        return {"values": dict(values), "reasons": dict(reasons.most_common())}

    has_gates = any(a["upstream_gates"] for a in attempts)
    for name, field, reasons_field in (
        ("camera_court", "court_valid", "court_reasons"),
        ("tracking", "tracking", None),
        ("point_validity", "validity", "validity_reasons"),
    ):
        counts = gate_counts(field, reasons_field) if has_gates else {}
        top = next(iter(counts.get("reasons") or {}), None)
        stages[name] = {
            "status": "ok" if has_gates else "not_recorded",
            "reason": top,
            "counts": counts,
        }
    fit_status = Counter(a["first_fit"]["status"] for a in attempts)
    held = Counter(
        a["first_fit"]["reason"] for a in attempts if a["first_fit"]["status"] != "completed"
    )
    event_held = {k: v for k, v in held.items() if k and k.startswith("events_")}
    stages["events"] = {
        "status": steps["upstream"]["status"],
        "reason": next(iter(Counter(event_held).most_common(1)), (None,))[0],
        "counts": {"attempts_refused_for_events": sum(event_held.values()), **event_held},
    }
    components = Counter(c["reason"] or c["status"] for a in attempts for c in a["components"])
    flights = Counter(f["reason"] for a in attempts for f in a["flights"])
    stages["s6_fit"] = {
        "status": steps["fit"]["status"],
        "reason": steps["fit"].get("reason"),
        "counts": {
            "attempts": len(attempts),
            "by_status": dict(fit_status),
            "held_or_failed": dict(held.most_common()),
            "components": dict(components.most_common()),
            "flight_slots": dict(flights.most_common()),
        },
    }
    for step in ("cascade", "refit", "assemble", "portal", "compact"):
        stages[step] = {
            "status": steps[step]["status"],
            "reason": steps[step].get("reason"),
            "counts": steps[step].get("detail") or {},
        }
    calls = Counter(
        (a.get("point_end") or {}).get("evidence", "none").split(":")[0]
        for a in attempts
        if a["accepted_flights"]
    )
    stages["post_point_tag"] = {
        "status": steps["assemble"]["status"],
        "reason": None,
        "counts": {
            "point_end_calls": dict(calls),
            "accepted_flights_after_point_end": sum(
                a.get("accepted_flights_after_point_end") or 0 for a in attempts
            ),
        },
    }
    order = (
        "source_admission",
        "point_segmentation",
        "camera_court",
        "tracking",
        "point_validity",
        "events",
        "s6_fit",
        "cascade",
        "refit",
        "post_point_tag",
        "assemble",
        "portal",
        "compact",
    )
    return {name: stages[name] for name in order}


def summary_counts(attempts: list[dict]) -> dict:
    return {
        "attempts": len(attempts),
        "attempts_with_flights": sum(bool(a["accepted_flights"]) for a in attempts),
        "complete_points": sum(
            a["complete_point"] and bool(a["accepted_flights"]) for a in attempts
        ),
        "accepted_flights": sum(a["accepted_flights"] for a in attempts),
        "flight_slots": sum(a["slots"] or 0 for a in attempts),
        "by_reason": dict(Counter(a["reason"] for a in attempts).most_common()),
    }


def failure_point(stages: dict, steps: dict) -> dict | None:
    for step in STEPS:
        if steps[step]["status"] == "failed":
            stage = (
                "source_admission" if stages["source_admission"]["status"] == "refused" else step
            )
            return {
                "step": step,
                "stage": stage,
                "reason": stages["source_admission"]["reason"]
                if stage == "source_admission"
                else steps[step]["reason"],
            }
    return None


# --------------------------------------------------------------------------- build


def build(
    out: Path,
    match_id: str,
    *,
    attempts: list[dict] | None = None,
    video: str | None = None,
    policy: dict | None = None,
    code: dict | None = None,
    extra: dict | None = None,
) -> dict:
    """The outcome document of one match directory from what is on disk.

    ``attempts`` are the per-attempt records (:func:`attempt_outcome`); ``None`` keeps the
    attempts of an existing ``OUTCOMES.json`` (the fit trees may be archived by then).
    """
    out = Path(out)
    previous = _load(out / FILE) if (out / FILE).is_file() else {}
    if attempts is None:
        attempts = relocate(out, previous.get("attempts") or [])
    match_upstream = out / "upstream" / match_id
    if video is None:
        item = out / "QUEUE_ITEM.json"
        video = (
            _load(item).get("video") if item.is_file() else previous.get("source", {}).get("video")
        )
    source = source_record(match_upstream, video)
    if not source.get("cadence_decision") and previous.get("source"):
        source = {**previous["source"], **{k: v for k, v in source.items() if v is not None}}
    if (out / "QUEUE_ITEM.json").is_file():
        # Partial or highlights videos stay out of per-match statistics (product_queue).
        source["partial_or_highlights"] = bool(
            _load(out / "QUEUE_ITEM.json").get("partial_or_highlights")
        )
    identity = dict(previous.get("identity") or {})
    if (match_upstream / "run_manifests" / "stage_receipts").is_dir():
        identity.update(upstream_identity(match_upstream))
    if policy is not None or "s6_policy_sha256" not in identity:
        identity.update(policy_identity(policy))
    if code is not None:
        identity["code"] = code
    identity.setdefault("code", provenance.git_record(REPO))
    decisions = out / "cascade" / "adjudicate" / "decisions.json"
    if decisions.is_file():
        settings = _load(decisions).get("settings") or {}
        identity["event_adjudication"] = settings.get("adjudication", "paid")
    #: Fields callers add (``queue_failure``), kept across later refreshes.
    kept = {name: previous[name] for name in previous.get("extra_fields") or [] if name in previous}
    kept.update(extra or {})
    identity_file = out / player_names.FILE
    player_identity = None
    if identity_file.is_file():
        # Near/far, server and source per attempt (player_names, written by ``portal``).
        names_report = _load(identity_file)
        player_identity = player_names.summary(names_report)
        for attempt in attempts:
            number = player_names.attempt_number(attempt.get("key", ""))
            attempt["player_identity"] = player_names.identity_record(
                names_report["attempts"].get(str(number))
            )
    steps = step_rows(out)
    stages = stage_rows(out, match_id, source, attempts, steps)
    done = all(steps[s]["status"] == "done" for s in STEPS)
    failure = failure_point(stages, steps)
    queue_failure = kept.get("queue_failure")
    if failure is None and queue_failure and not done:
        failure = {"step": queue_failure.get("step"), "stage": "queue", **queue_failure}
    status = "failed" if failure else ("done" if done else "incomplete")
    document = {
        "schema": SCHEMA,
        "match_id": match_id,
        "built": time.time(),
        "builder": provenance.git_record(REPO),
        "status": status,
        "failure": failure,
        "identity": identity,
        "source": source,
        "steps": steps,
        "stages": stages,
        # What this match ran differently than asked: {switch: [kinds]} over every step.
        "switch_fallbacks": switch_fallbacks.merge(
            {k: v.get("fallbacks") for k, v in steps.items()}
        ),
        "switches": step_switches(out),
        "counts": summary_counts(attempts),
        **({"player_identity": player_identity} if player_identity else {}),
        "yield_note": (
            "Automatic output yield (attempts with accepted flights / attempts); no labels "
            "exist for these broadcasts, so no correctness is claimed."
        ),
        "evaluation_labels_read": False,
        "extra_fields": sorted(kept),
        **kept,
        "attempts": attempts,
    }
    return document


def relocate(out: Path, attempts: list[dict]) -> list[dict]:
    """Point each retained fit job at where it is now (tree, archive after compact, gone)."""
    archive = (Path(out) / S6_ARCHIVE).is_file()
    for attempt in attempts:
        record = attempt.get("reprocess") or {}
        if not record.get("fit_job"):
            continue
        if (Path(out) / record["fit_job"]).is_file():
            record.update(location="tree", available=True)
        elif archive:
            record.update(location=S6_ARCHIVE, available=True)
        else:
            record.update(location=None, available=False)
    return attempts


def write(out: Path, match_id: str, **kwargs) -> dict:
    document = build(out, match_id, **kwargs)
    _save(Path(out) / FILE, document)
    return document


def refresh(out: Path, match_id: str) -> dict | None:
    """Re-read step markers into an existing record (after portal, compact, a failure)."""
    try:
        return write(out, match_id)
    except Exception as error:  # noqa: BLE001 - an outcome record never fails a match
        return {"error": f"{type(error).__name__}: {error}"}


# --------------------------------------------------------------------------- from disk


class StageFiles:
    """Read S6 stage files of a match directory, from the tree or its verified archive."""

    PATTERNS = (
        "fit/summary.json",
        "fit/attempts/*/stage/result.json",
        "fit/attempts/*/stage_run/cases/*/result.json",
        "fit/attempts/*/child_*/stage_run/cases/*/result.json",
        "fit/attempts/*/job.json",
        "fit/attempts/*/child_*/job.json",
        "refit/*/stage/result.json",
        "refit/*/stage_run/cases/*/result.json",
        "refit/REFIT.json",
        # Lean receipts (TENNIS_S6_LEAN): a case result's evaluation context sits beside it.
        "fit/attempts/*/stage_run/cases/*/evaluation_context.json",
        "fit/attempts/*/child_*/stage_run/cases/*/evaluation_context.json",
        "refit/*/stage_run/cases/*/evaluation_context.json",
    )

    def __init__(self, out: Path) -> None:
        self.out = Path(out)
        self._scratch: tempfile.TemporaryDirectory | None = None
        self.root = self.out
        self.location = "tree"
        archive = self.out / S6_ARCHIVE
        if archive.is_file() and not (self.out / "fit" / "attempts").is_dir():
            self._scratch = tempfile.TemporaryDirectory(prefix="outcomes_", dir=self.out.parent)
            subprocess.run(
                [
                    "tar",
                    "-I",
                    UNZSTD,
                    "-xf",
                    str(archive),
                    "-C",
                    self._scratch.name,
                    "--wildcards",
                    "--no-wildcards-match-slash",
                    *self.PATTERNS[1:],
                ],
                check=False,
                capture_output=True,
            )
            self.root = Path(self._scratch.name)
            self.location = S6_ARCHIVE

    def result(self, stage: Path) -> dict | None:
        """``stage`` as the runner names it (``fit/attempts/<key>/stage``, may be a link)."""
        relative = Path(stage).relative_to(self.out) if Path(stage).is_absolute() else Path(stage)
        from cv.pipeline import s6_lean_receipts

        direct = self.root / relative / "result.json"
        if direct.is_file():
            return s6_lean_receipts.inline(_load(direct), direct.parent)
        parent = self.root / relative.parent
        found = sorted(parent.glob("stage_run/cases/*/result.json"))
        return s6_lean_receipts.inline(_load(found[0]), found[0].parent) if found else None

    def job(self, relative: str) -> Path | None:
        path = self.root / relative
        return path if path.is_file() else None

    def close(self) -> None:
        if self._scratch is not None:
            self._scratch.cleanup()


def fit_job_record(files: StageFiles, attempt: dict, out: Path) -> dict:
    """Where the attempt's retained S6 fit inputs are (``job.json`` + ``inputs/``)."""
    job = Path(attempt["job"])
    relative = str(job.relative_to(out)) if job.is_absolute() else str(job)
    present = files.job(relative)
    if present is None:
        return {"fit_job": None, "available": False, "needs": "s6_prepare_fit"}
    return {
        "fit_job": relative,
        "location": files.location,
        "available": True,
        "needs": "s6_fit",
        "job_sha256": provenance.file_sha256(present),
    }


def attempts_from_disk(out: Path, match_id: str) -> tuple[list[dict], dict | None]:
    """Backfill: every attempt record of a finished (possibly compacted) match directory."""
    from cv.pipeline import point_end_tag
    from cv.pipeline.product_runner import attempt_flights

    out = Path(out)
    summary_path = out / "fit" / "summary.json"
    if not summary_path.is_file():
        return [], None
    summary = _load(summary_path)
    files = StageFiles(out)
    try:
        refits = {}
        refit_path = files.root / "refit" / "REFIT.json"
        if not refit_path.is_file():
            refit_path = out / "refit" / "REFIT.json"
        if refit_path.is_file():
            refits = {r["attempt"]: r for r in _load(refit_path)["runs"]}
        gates = upstream_gates(out / "upstream")
        records, policy = [], None
        for row in summary["attempts"]:
            folder = Path("fit") / "attempts" / row["key"]
            members = (
                [(child, folder / f"child_{child['child_index']:02d}") for child in row["children"]]
                if row.get("children")
                else [(row, folder)]
            )
            for member, place in members:
                attempt = {
                    "key": member["key"],
                    "parent_key": row["key"],
                    "clip": row["clip"],
                    "status": member["status"],
                    "reason": member.get("reason"),
                    "ledger_row": row.get("ledger_row"),
                    "job": place / "job.json",
                }
                refit = refits.get(member["key"])
                result, source = None, None
                if refit and refit.get("status") == "completed":
                    result = files.result(Path("refit") / member["key"] / "stage")
                    source = "cascade_refit" if result else None
                if result is None and member["status"] == "completed":
                    result = files.result(place / "stage")
                    source = "first_fit" if result else None
                flights = attempt_flights(result) if result else []
                point_end = point_end_tag.tag_result(result, flights) if result else None
                if policy is None and files.job(str(place / "job.json")):
                    policy = _load(files.job(str(place / "job.json"))).get("policy")
                records.append(
                    attempt_outcome(
                        attempt,
                        result,
                        final_source=source,
                        refit_status=refit.get("status") if refit else None,
                        final_flights=flights,
                        point_end=point_end,
                        gates=gates.get(row["clip"]),
                        fit_job=fit_job_record(files, attempt, out),
                    )
                )
        return records, policy
    finally:
        files.close()


def backfill(out: Path, match_id: str, destination: Path | None = None, **kwargs) -> dict:
    """Build an outcome record for a match directory written before outcomes existed.

    ``destination`` (a copy outside a read-only match directory): the record is written
    there and ``source_dir`` records where the match lives.
    """
    out = Path(out)
    attempts, policy = attempts_from_disk(out, match_id)
    code = None
    result_path = out / "MATCH_RESULT.json"
    if result_path.is_file():
        match_result = _load(result_path)
        code = match_result.get("code")
        if "video" not in kwargs and not (out / "QUEUE_ITEM.json").is_file():
            kwargs["video"] = (match_result.get("source_video") or {}).get("path")
    upstream_archive = out / UPSTREAM_ARCHIVE
    scratch = None
    if (
        upstream_archive.is_file()
        and not (out / "upstream" / "court_geometry_point_gate_v1.json").is_file()
    ):
        scratch = tempfile.TemporaryDirectory(
            prefix="outcomes_up_", dir=(destination or out).parent
        )
        subprocess.run(
            [
                "tar",
                "-I",
                UNZSTD,
                "-xf",
                str(upstream_archive),
                "-C",
                scratch.name,
                "--wildcards",
                "--no-wildcards-match-slash",
                *(f"upstream/{name}" for name in UPSTREAM_GATES),
                f"upstream/{match_id}/run_manifests/stage_receipts/*.json",
            ],
            check=False,
            capture_output=True,
        )
        gates = upstream_gates(Path(scratch.name) / "upstream")
        for attempt in attempts:
            attempt["upstream_gates"] = gates.get(attempt["clip"]) or attempt["upstream_gates"]
    try:
        document = build(out, match_id, attempts=attempts, policy=policy, code=code, **kwargs)
        if scratch is not None:
            receipts = Path(scratch.name) / "upstream" / match_id
            if (receipts / "run_manifests" / "stage_receipts").is_dir():
                document["identity"].update(upstream_identity(receipts))
    finally:
        if scratch is not None:
            scratch.cleanup()
    document["source_dir"] = str(out)
    document["backfilled"] = True
    _save((destination or out) / FILE, document)
    return document


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("match_dir", type=Path)
    parser.add_argument("--match-id", help="default: the directory name")
    parser.add_argument(
        "--destination", type=Path, help="write here instead of into the match directory"
    )
    args = parser.parse_args()
    match_id = args.match_id or args.match_dir.name
    document = backfill(args.match_dir.resolve(), match_id, args.destination)
    print(json.dumps({"status": document["status"], **document["counts"]}, indent=1))


if __name__ == "__main__":
    main()
