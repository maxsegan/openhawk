"""Run the paid event cascade between the automatic S6 fit and its second fit.

    python -m cv.pipeline.event_cascade_runner candidates --wave WAVE.json --run-root RUNS --out OUT [--adjudication paid|fitter] [--candidate-scope SCOPE]
    python -m cv.pipeline.event_cascade_runner adjudicate --out OUT [--adjudication paid|fitter] [--qwen-ports 8523,8524] [--cap-usd 10] [--first-tier qwen|gemini]
    python -m cv.pipeline.event_cascade_runner propose --wave WAVE.json --run-root RUNS --out OUT [--cap-usd 10]
    python -m cv.pipeline.event_cascade_runner apply --wave WAVE.json --out OUT --wave-name NAME [--propose off|retime|retime_locate]

``WAVE.json`` is the batch of automatic (AA) S6 fits and ``RUNS`` the
directory ``scripts/fleet_fits.py`` wrote them to. ``candidates`` reads each
fit's ``component_plan.json``; ``adjudicate`` renders evidence, asks the tiers
and writes ``decisions.json`` (every call's model, tokens and dollars) and
``COST.json``; ``apply`` writes the edited packets and ``CASCADE_WAVE.json``
for the second fit. Attempts with no settled decision are not refitted: their
first fit stands. Switches: ``--second-draw settle_only|always`` and
``--opus-evidence 1280|full`` (defaults are the cheaper measured policy), and
``--first-tier qwen|gemini`` (``gemini`` drops the local Qwen tier).
``--adjudication fitter`` asks no model: every candidate (scope
``event_cascade.FITTER_CANDIDATE_SCOPE`` unless given) is confirmed with provenance
``fitter_adjudicated`` at $0 and the refit's physics gates decide; ``paid`` is the
model cascade (rollback).
``propose`` (optional, ``event_cascade_propose``) opens label-free locate/retime windows,
asks Gemini Flash and Opus for an impact frame and writes ``propose/decisions.json`` and
``PROPOSE_COST.json``; ``apply --propose`` adds those edits (default ``off``: none).
No evaluation label is read; ``apply`` refuses a packet whose ball or events
are not automatic.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cv.pipeline import event_cascade as cascade
from cv.pipeline import event_cascade_models as models
from cv.pipeline import event_cascade_propose as proposer
from cv.pipeline import event_cascade_rebind as rebind

DATA = Path(os.environ.get("TENNIS_DATA_ROOT", "data"))
ATTEMPTS_PER_MATCH = 230


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _resolve(record: dict) -> Path:
    if record.get("path_base") != "TENNIS_DATA_ROOT":
        raise ValueError(f"path base {record.get('path_base')}")
    return DATA / record["path"]


def aa_jobs(wave: dict, panels: tuple[str, ...] | None = None) -> list[dict]:
    return [
        job
        for job in wave["jobs"]
        if job.get("input_arm") == "AA" and (not panels or job.get("panel") in panels)
    ]


def _plan_path(run_root: Path, name: str) -> Path | None:
    hits = sorted(run_root.glob(f"*/runs/{name}/cases/*/component_plan.json"))
    if len(hits) > 1:
        raise ValueError(f"two component plans for {name}")
    return hits[0] if hits else None


def build_candidates(
    jobs: list[dict], run_root: Path | None = None, scope: str = "unresolved", *, plan_for=None
) -> dict:
    """Candidates of every fitted attempt. A fit with no component plan adds none.

    ``plan_for(job)`` names a job's ``component_plan.json`` (or None); the default looks
    it up under the batch ``run_root`` layout.
    """
    if scope not in cascade.CANDIDATE_SCOPES:
        raise ValueError(f"candidate scope must be one of {cascade.CANDIDATE_SCOPES}")
    rows, attempts, missing = [], [], []
    for job in jobs:
        path = plan_for(job) if plan_for is not None else _plan_path(run_root, job["name"])
        if path is None:
            missing.append({"name": job["name"], "attempt": job["attempt"]})
            continue
        plan = _load(path)
        pose = _resolve(_load(Path(job["manifest"]))["rows"][0]["pose_csv"])
        verdict = None if scope == "unresolved" else _load(path.with_name("result.json"))["verdict"]
        chosen = cascade.candidate_rows(
            job, plan, cascade.match_directory(pose), scope=scope, verdict=verdict
        )
        attempts.append(
            {"panel": job.get("panel"), "attempt": job["attempt"], "candidates": len(chosen)}
        )
        rows.extend(chosen)
    return {
        "schema": cascade.CANDIDATES_SCHEMA,
        "freeze": cascade.SCHEMA,
        "rule": (
            "A decoder-emitted unsupported occurrence that bounds or sits inside a flight "
            "the automatic plan did not accept. Labels are not an input."
        ),
        "candidate_scope": scope,
        "wave_attempts": len(jobs),
        "fitted_attempts": len(attempts),
        "missing_plans": missing,
        "candidates": len(rows),
        "bands": dict(Counter(row["band"] for row in rows)),
        "attempts": attempts,
        "rows": rows,
    }


def contradictions(rows: list[dict]) -> dict[str, bool]:
    tracks: dict[tuple[str, str], dict] = {}
    found = {}
    for row in rows:
        key = (row["track_csv"], row["point"])
        if key not in tracks:
            tracks[key] = cascade.track_points(Path(row["track_csv"]), row["point"])
        found[row["id"]] = cascade.contradicts_track(row, tracks[key])
    return found


def walk_all(
    rows: list[dict], answer_for, contradicts: dict[str, bool], chosen: dict
) -> dict[str, dict]:
    """``answer_for(row, tier, draw)``: the saved answer or None."""
    return {
        row["id"]: cascade.settle(
            row,
            lambda tier, draw, row=row: answer_for(row, tier, draw),
            contradicts=contradicts[row["id"]],
            chosen=chosen,
        )
        for row in rows
    }


def decision_document(rows: list[dict], walks: dict, contradicts: dict, chosen: dict) -> dict:
    decisions, pending = [], 0
    for row in rows:
        walk = walks[row["id"]]
        if walk["owed"] is not None:
            pending += 1
            continue
        decisions.append(
            {
                "id": row["id"],
                "attempt": row["attempt"],
                "panel": row["panel"],
                "clip": row["clip"],
                "event_index": row["event_index"],
                "event_type": row["event_type"],
                "frame": row["frame"],
                "band": row["band"],
                "contradicts_track": contradicts[row["id"]],
                "action": walk["action"],
                "tier": walk["tier"],
                "model": walk["model"],
                "receipts": walk["receipts"],
            }
        )
    receipts = [item for decision in decisions for item in decision["receipts"]]
    by_model: dict[str, dict] = defaultdict(
        lambda: {"calls": 0, "usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
    )
    for item in receipts:
        entry = by_model[item["model"]]
        entry["calls"] += 1
        entry["usd"] += item["usd"]
        entry["prompt_tokens"] += item["tokens"].get("prompt") or 0
        entry["completion_tokens"] += item["tokens"].get("completion") or 0
    if chosen.get("adjudication") == "fitter":
        rule = (
            "Fitter adjudication: every candidate is confirmed as a supported automatic event "
            f"({cascade.FITTER_SOURCE}); no model is asked. The refit's physics gates accept or "
            "reject the flights it bounds."
        )
    else:
        rule = (
            ("" if chosen["first_tier"] == "qwen" else "No Qwen tier (first_tier=gemini). ")
            + "Qwen guided_nothink, then Gemini zero, then Opus zero. First tier whose draws all "
            "confirm the emitted type within 2 frames, or all reject, settles. A reject that "
            "contradicts the ball track climbs and after Opus stays unsettled. Unsettled rows are "
            "not edited."
        )
    return {
        "schema": cascade.DECISIONS_SCHEMA,
        "freeze": "v2",
        "settings": chosen,
        "rule": rule,
        "candidates": len(rows),
        "decided": len(decisions),
        "pending": pending,
        "actions": dict(Counter(decision["action"] for decision in decisions)),
        "settled_by_tier": dict(
            Counter(d["tier"] for d in decisions if d["action"] != "unsettled")
        ),
        "receipt_usd": round(sum(item["usd"] for item in receipts), 6),
        "by_model": {
            model: dict(value, usd=round(value["usd"], 6)) for model, value in by_model.items()
        },
        "decisions": decisions,
    }


def _run_candidates(args) -> None:
    wave = _load(args.wave)
    scope = cascade.candidate_scope_for(args.adjudication, args.candidate_scope)
    document = build_candidates(aa_jobs(wave, _panels(args)), Path(args.run_root), scope)
    document["wave"] = str(args.wave)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates.json").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({key: document[key] for key in ("fitted_attempts", "candidates", "bands")}))


def _run_adjudicate(args) -> None:
    if args.adjudication == "fitter":
        fitter_adjudicate(Path(args.out))
        return
    adjudicate(
        Path(args.out),
        qwen_ports=args.qwen_ports,
        cap_usd=args.cap_usd,
        second_draw=args.second_draw,
        opus_evidence=args.opus_evidence,
        first_tier=args.first_tier,
    )


def adjudicate(
    out: Path,
    *,
    qwen_ports: str,
    cap_usd: float,
    second_draw: str = cascade.DEFAULT_SETTINGS["second_draw"],
    opus_evidence: str = cascade.DEFAULT_SETTINGS["opus_evidence"],
    first_tier: str = cascade.DEFAULT_SETTINGS["first_tier"],
) -> tuple[dict, dict]:
    """Walk every candidate in ``out/candidates.json`` to a decision; write decisions and cost."""
    chosen = cascade.settings(
        {"second_draw": second_draw, "opus_evidence": opus_evidence, "first_tier": first_tier}
    )
    rows = _load(out / "candidates.json")["rows"]
    root = out / "adjudicate"
    ledger = root / "spend.jsonl"
    evidence = root / "evidence" / "one" / "pergrid"
    contradicts = contradictions(rows)
    rendered = {row["id"]: models.render(row, evidence) for row in rows}
    ports = tuple(int(port) for port in qwen_ports.split(",") if port.strip())
    demos = None
    qwen_seconds = 0.0
    qwen_calls = 0

    def answer_for(row, tier, draw):
        return cascade.read_answer(models.reply_path(root, tier, draw, row, chosen))

    failures: list[str] = []
    for _round in range(8):
        walks = walk_all(rows, answer_for, contradicts, chosen)
        owed = [
            (row, *walks[row["id"]]["owed"]) for row in rows if walks[row["id"]]["owed"] is not None
        ]
        if not owed or failures:
            break
        local = [item for item in owed if item[1] == "qwen"]
        hosted = [item for item in owed if item[1] != "qwen"]
        if local:
            if demos is None:
                models.wait_ready(ports)
                demos = models.qwen_demos()
            started = time.time()

            def ask_local(indexed):
                index, (row, tier, draw) = indexed
                base = f"http://127.0.0.1:{ports[index % len(ports)]}/v1"
                return models.call(
                    row,
                    rendered[row["id"]],
                    tier,
                    draw,
                    root=root,
                    chosen=chosen,
                    ledger=ledger,
                    cap_usd=cap_usd,
                    base_url=base,
                    demos=demos,
                )

            with ThreadPoolExecutor(models.QWEN_PER_PORT * len(ports)) as pool:
                for line in pool.map(ask_local, enumerate(local)):
                    print(line, flush=True)
                    failures += [line] if line.startswith("FAIL") else []
            qwen_seconds += len(ports) * (time.time() - started)
            qwen_calls += len(local)
        if hosted:

            def ask_hosted(item):
                row, tier, draw = item
                return models.call(
                    row,
                    rendered[row["id"]],
                    tier,
                    draw,
                    root=root,
                    chosen=chosen,
                    ledger=ledger,
                    cap_usd=cap_usd,
                )

            with ThreadPoolExecutor(models.HOSTED_WORKERS) as pool:
                for line in pool.map(ask_hosted, hosted):
                    print(line, flush=True)
                    failures += [line] if line.startswith("FAIL") else []
    walks = walk_all(rows, answer_for, contradicts, chosen)
    document = decision_document(rows, walks, contradicts, chosen)
    document["spend_ledger_usd"] = round(models.spent(ledger), 6)
    (root / "decisions.json").write_text(json.dumps(document, indent=2) + "\n")
    saved = _load(out / "candidates.json")
    attempts = (
        saved.get("wave_attempts")
        or (len(saved.get("attempts") or []) + len(saved.get("missing_plans") or []))
        or None
    )
    cost = cost_summary(
        document, attempts=attempts, qwen_gpu_seconds=qwen_seconds, qwen_new_calls=qwen_calls
    )
    (out / "COST.json").write_text(json.dumps(cost, indent=2) + "\n")
    print(
        json.dumps({key: document[key] for key in ("decided", "pending", "actions", "receipt_usd")})
    )
    if failures or document["pending"]:
        raise SystemExit(
            f"cascade incomplete: {len(failures)} failed calls, {document['pending']} pending"
        )
    return document, cost


def fitter_adjudicate(out: Path, *, attempts: int | None = None) -> tuple[dict, dict]:
    """Confirm every candidate in ``out/candidates.json`` without a model; $0 cost file."""
    saved = _load(out / "candidates.json")
    rows = saved["rows"]
    walks = {row["id"]: cascade.fitter_settle(row) for row in rows}
    document = decision_document(
        rows,
        walks,
        {row["id"]: None for row in rows},
        {"adjudication": "fitter", "candidate_scope": saved.get("candidate_scope")},
    )
    document["spend_ledger_usd"] = 0.0
    root = out / "adjudicate"
    root.mkdir(parents=True, exist_ok=True)
    (root / "decisions.json").write_text(json.dumps(document, indent=2) + "\n")
    if attempts is None:
        attempts = (
            saved.get("wave_attempts")
            or (len(saved.get("attempts") or []) + len(saved.get("missing_plans") or []))
            or None
        )
    cost = cost_summary(document, attempts=attempts, qwen_gpu_seconds=0.0, qwen_new_calls=0)
    (out / "COST.json").write_text(json.dumps(cost, indent=2) + "\n")
    print(json.dumps({key: document[key] for key in ("decided", "actions", "receipt_usd")}))
    return document, cost


def cost_summary(
    document: dict, *, attempts: int | None, qwen_gpu_seconds: float, qwen_new_calls: int
) -> dict:
    """Dollars as bought (receipts) and Qwen GPU-seconds, per fitted attempt and per match."""
    usd = document["receipt_usd"]
    qwen_receipts = sum(
        1
        for decision in document["decisions"]
        for item in decision["receipts"]
        if item["tier"] == 1
    )
    return {
        "wave_attempts": attempts,
        "receipt_usd": usd,
        "by_model": document["by_model"],
        "qwen_calls_in_receipts": qwen_receipts,
        "qwen_new_calls": qwen_new_calls,
        "qwen_gpu_seconds_new": round(qwen_gpu_seconds, 1),
        "usd_per_attempt": round(usd / attempts, 6) if attempts else None,
        "usd_per_match": round(usd / attempts * ATTEMPTS_PER_MATCH, 3) if attempts else None,
        "note": (
            "Denominator is every automatic attempt in the fit wave, with or without a candidate. "
            f"A match is {ATTEMPTS_PER_MATCH} attempts. Reused replies count at what they cost when bought."
        ),
    }


def propose_windows(
    jobs: list[dict],
    run_root: Path | None = None,
    *,
    plan_for=None,
    settled: list[dict] | None = None,
) -> dict:
    """Locate and retime windows of every fitted attempt whose scope the cascade can edit.

    ``settled`` are the FREEZE v2 decisions: a row they confirm or reject is not retimed,
    so its window is not asked."""
    done = {
        (d["panel"], d["attempt"], round(float(d["frame"]), 5))
        for d in settled or []
        if d.get("action") in ("confirm", "reject") and d.get("event_type") == "contact"
    }
    windows, skipped = [], []
    for job in jobs:
        path = plan_for(job) if plan_for is not None else _plan_path(run_root, job["name"])
        if path is None:
            skipped.append({"attempt": job["attempt"], "why": "no_component_plan"})
            continue
        plan = _load(path)
        verdict = _load(path.with_name("result.json"))["verdict"]
        manifest = _load(Path(job["manifest"]))
        row = manifest["rows"][0]
        packet = _load(_resolve(row["packet"]))
        inventory = proposer.editable_inventory(packet)
        if inventory is None:
            skipped.append({"attempt": job["attempt"], "why": "uneditable_scope"})
            continue
        labels = _load(_resolve(row["labels"]))
        ball = [
            float(frame["frame"])
            for record in labels["ball"]["records"]
            for frame in record["frames"]
            if frame.get("status") == "visible"
        ]
        match = cascade.match_directory(_resolve(row["pose_csv"]))
        found = proposer.attempt_windows(
            job=job,
            plan=plan,
            events=inventory,
            ball_frames=ball,
            verdict=verdict,
            fps=float(packet["attempts"][0]["fps"]),
            match=match,
            frames_dir=str(match / cascade.FRAMES / plan["clip"]),
            track_csv=str(match / cascade.TRACK),
        )
        for window in found:
            key = (job.get("panel"), job["attempt"], round(float(window["event_frame"] or 0), 5))
            if window["mode"] == "retime" and key in done:
                skipped.append({"window": window["id"], "why": "cascade_settled_row"})
                continue
            windows.append(window)
    return {
        "schema": proposer.SCHEMA,
        "rule": (
            "Label-free: contact-to-contact gaps and leading stretches not covered by an "
            "accepted flight (locate), automatic contacts bounding a rejected flight (retime)."
        ),
        "wave_attempts": len(jobs),
        "skipped": skipped,
        "windows": windows,
    }


def run_propose(out: Path, *, cap_usd: float, workers: int = models.HOSTED_WORKERS) -> dict:
    """Ask every window in ``out/propose/windows.json`` to a decision; write decisions and cost."""
    root = out / "propose"
    saved = _load(root / "windows.json")
    windows = saved["windows"]
    ledger = root / "spend.jsonl"
    evidence = root / "evidence"
    renders: dict[tuple[str, str], dict] = {}

    def nom(window, presentation):
        key = (window["id"], presentation)
        if key not in renders:
            renders[key] = proposer.render(window, presentation, evidence)
        return renders[key]

    def answer_for(window, tier, presentation):
        return proposer.read(proposer.reply_path(root, tier, presentation, window))

    failures: list[str] = []
    for _round in range(4):
        walks = {
            w["id"]: proposer.settle(w, lambda t, p, w=w: answer_for(w, t, p)) for w in windows
        }
        owed = [(w, *walks[w["id"]]["owed"]) for w in windows if walks[w["id"]]["owed"]]
        if not owed or failures:
            break

        def ask(item):
            window, tier, presentation = item
            return proposer.call(
                window,
                nom(window, presentation),
                tier,
                presentation,
                root=root,
                ledger=ledger,
                cap_usd=cap_usd,
            )

        with ThreadPoolExecutor(workers) as pool:
            for line in pool.map(ask, owed):
                print(line, flush=True)
                failures += [line] if line.startswith("FAIL") else []
    walks = {w["id"]: proposer.settle(w, lambda t, p, w=w: answer_for(w, t, p)) for w in windows}
    decisions, pending = [], 0
    for window in windows:
        walk = walks[window["id"]]
        if walk["owed"]:
            pending += 1
            continue
        decisions.append(
            {
                **window,
                "window": window["id"],
                "action": walk["action"],
                "frame": walk.get("frame"),
                "tier": walk["tier"],
                "model": walk.get("model"),
                "receipts": walk["receipts"],
            }
        )
    receipts = [item for decision in decisions for item in decision["receipts"]]
    by_model: dict[str, dict] = defaultdict(
        lambda: {"calls": 0, "usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
    )
    for item in receipts:
        entry = by_model[item["model"]]
        entry["calls"] += 1
        entry["usd"] += item["usd"]
        entry["prompt_tokens"] += item["tokens"].get("prompt") or 0
        entry["completion_tokens"] += item["tokens"].get("completion") or 0
    usd = round(sum(item["usd"] for item in receipts), 6)
    document = {
        "schema": proposer.DECISIONS_SCHEMA,
        "rule": (
            "locate: Flash ball and Flash player; both no_contact settles none, else one Opus "
            "ball_player draw proposes a contact inside the window. retime: one Opus ball draw; a "
            f"contact >= {proposer.RETIME_MIN} frames from the automatic epoch moves it."
        ),
        "windows": len(windows),
        "decided": len(decisions),
        "pending": pending,
        "actions": dict(Counter(f"{d['mode']}:{d['action']}" for d in decisions)),
        "receipt_usd": usd,
        "spend_ledger_usd": round(models.spent(ledger), 6),
        "by_model": {m: dict(v, usd=round(v["usd"], 6)) for m, v in by_model.items()},
        "decisions": decisions,
    }
    (root / "decisions.json").write_text(json.dumps(document, indent=2) + "\n")
    attempts = saved.get("wave_attempts")
    by_mode = defaultdict(float)
    for decision in decisions:
        by_mode[decision["mode"]] += sum(item["usd"] for item in decision["receipts"])
    cost = {
        "wave_attempts": attempts,
        "windows": len(windows),
        "receipt_usd": usd,
        "by_mode_usd": {k: round(v, 6) for k, v in by_mode.items()},
        "by_model": document["by_model"],
        "usd_per_match": round(usd / attempts * ATTEMPTS_PER_MATCH, 3) if attempts else None,
        "retime_usd_per_match": (
            round(by_mode["retime"] / attempts * ATTEMPTS_PER_MATCH, 3) if attempts else None
        ),
        "note": f"Denominator is every automatic attempt in the fit wave. A match is {ATTEMPTS_PER_MATCH} attempts.",
    }
    (out / "PROPOSE_COST.json").write_text(json.dumps(cost, indent=2) + "\n")
    print(
        json.dumps(
            {k: document[k] for k in ("windows", "decided", "pending", "actions", "receipt_usd")}
        )
    )
    if failures or pending:
        raise SystemExit(f"propose incomplete: {len(failures)} failed calls, {pending} pending")
    return document


def _run_propose(args) -> None:
    out = Path(args.out)
    decided = out / "adjudicate" / "decisions.json"
    settled = _load(decided)["decisions"] if decided.is_file() else None
    document = propose_windows(
        aa_jobs(_load(args.wave), _panels(args)), Path(args.run_root), settled=settled
    )
    document["settled_from"] = str(decided) if settled is not None else None
    document["wave"] = str(args.wave)
    (out / "propose").mkdir(parents=True, exist_ok=True)
    (out / "propose" / "windows.json").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({"windows": len(document["windows"]), "skipped": len(document["skipped"])}))
    run_propose(out, cap_usd=args.cap_usd, workers=args.workers)


def build_wave(
    jobs: list[dict],
    decisions: list[dict],
    out: Path,
    wave_name: str,
    *,
    proposals: list[dict] | None = None,
    propose: str = proposer.DEFAULT_SWITCH,
) -> dict:
    """Edited manifests for attempts with a settled decision or propose edit, and the refit wave."""
    by_attempt: dict[tuple, list[dict]] = defaultdict(list)
    for decision in decisions:
        by_attempt[(decision["panel"], decision["attempt"])].append(decision)
    windows: dict[tuple, list[dict]] = defaultdict(list)
    for decision in proposals or []:
        windows[(decision["panel"], decision["attempt"])].append(decision)
    built, edited, dropped = [], [], []
    for job in jobs:
        panel, attempt = job.get("panel"), job["attempt"]
        editable = rebind.settling(by_attempt.get((panel, attempt), []))
        edits: list[dict] = []
        if propose != "off" and windows.get((panel, attempt)):
            manifest = _load(Path(job["manifest"]))
            packet = _load(_resolve(manifest["rows"][0]["packet"]))
            inventory = proposer.editable_inventory(packet)
            first = windows[(panel, attempt)][0]
            if inventory is None:
                dropped += [
                    {"id": d["id"], "why": "uneditable_scope"}
                    for d in windows[(panel, attempt)]
                    if d["action"] in ("insert", "retime")
                ]
            else:
                edits, lost = proposer.propose_edits(
                    windows[(panel, attempt)],
                    inventory,
                    editable,
                    switch=propose,
                    track_csv=Path(first["track_csv"]),
                    point=first["point"],
                )
                dropped += lost
        if not editable and not edits:
            continue
        manifest = _load(Path(job["manifest"]))
        row = manifest["rows"][0]
        packet, labels, cameras = (
            _load(_resolve(row[key])) for key in ("packet", "labels", "cameras")
        )
        new_packet, new_labels = rebind.rebind(packet, labels, cameras, editable, edits)
        manifest_out = rebind.write_manifest(
            manifest,
            new_packet,
            new_labels,
            out / "packets" / f"{panel}__{attempt}",
            editable,
            edits,
        )
        built.append(
            {
                "name": f"CASCADE__{panel}__{attempt}__AA",
                "pair": f"cascade:{panel}:{attempt}:AA",
                "arm": "cascade",
                "panel": panel,
                "source": job.get("source"),
                "attempt": attempt,
                "input_arm": "AA",
                "fitter_pair": job.get("fitter_pair"),
                "manifest": str(manifest_out),
                "policy_overrides": job.get("policy_overrides") or {},
                "timeout_seconds": job.get("timeout_seconds", 4200),
                "search_seconds": job.get("search_seconds", 900),
            }
        )
        edited.append(
            {
                "panel": panel,
                "attempt": attempt,
                "decisions": sorted(
                    [f"{d['action']}:{d['event_type']}:{d['frame']}" for d in editable.values()]
                    + [proposer.edit_label(edit) for edit in edits]
                ),
            }
        )
    fitter = bool(decisions) and all(d.get("tier") == cascade.FITTER_SOURCE for d in decisions)
    document = {
        "schema": "event_cascade_wave_v1",
        "wave": wave_name,
        "freeze": "v2",
        "adjudication": "fitter" if fitter else "paid",
        "question": (
            "Automatic fits plus fitter adjudication: every candidate confirmed, no model asked; "
            "the refit's gates decide."
            if fitter
            else "Automatic fits plus the FREEZE v2 event cascade. Confirms support, rejects remove."
        ),
        "edited": edited,
        "jobs": built,
    }
    if propose != "off":
        document["propose"] = propose
        document["propose_dropped"] = dropped
    return document


def _run_apply(args) -> None:
    out = Path(args.out)
    saved = _load(out / "adjudicate" / "decisions.json")
    if saved.get("pending"):
        raise SystemExit(f"adjudication still pending: {saved['pending']}")
    proposals = None
    if args.propose != "off":
        proposed = _load(out / "propose" / "decisions.json")
        if proposed.get("pending"):
            raise SystemExit(f"propose still pending: {proposed['pending']}")
        proposals = proposed["decisions"]
    document = build_wave(
        aa_jobs(_load(args.wave), _panels(args)),
        saved["decisions"],
        out,
        args.wave_name,
        proposals=proposals,
        propose=args.propose,
    )
    (out / "CASCADE_WAVE.json").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({"edited": len(document["jobs"]), "wave": str(out / "CASCADE_WAVE.json")}))


def _panels(args) -> tuple[str, ...] | None:
    return tuple(item for item in (args.panels or "").split(",") if item) or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="step", required=True)
    for name in ("candidates", "adjudicate", "propose", "apply"):
        step = sub.add_parser(name)
        step.add_argument("--out", type=Path, required=True)
        step.add_argument("--panels", default="")
        if name in ("candidates", "propose", "apply"):
            step.add_argument("--wave", type=Path, required=True)
        if name in ("candidates", "propose"):
            step.add_argument("--run-root", type=Path, required=True)
        if name == "propose":
            step.add_argument("--cap-usd", type=float, default=10.0)
            step.add_argument("--workers", type=int, default=models.HOSTED_WORKERS)
        if name in ("candidates", "adjudicate"):
            step.add_argument(
                "--adjudication",
                choices=cascade.ADJUDICATION_CHOICES,
                default=cascade.DEFAULT_ADJUDICATION,
            )
        if name == "candidates":
            step.add_argument(
                "--candidate-scope",
                choices=cascade.CANDIDATE_SCOPES,
                default=None,
                help="default: the adjudication's scope (event_cascade.candidate_scope_for)",
            )
        if name == "adjudicate":
            step.add_argument("--qwen-ports", default="8523,8524")
            step.add_argument("--cap-usd", type=float, default=10.0)
            step.add_argument(
                "--second-draw", choices=cascade.SECOND_DRAW_CHOICES, default="settle_only"
            )
            step.add_argument(
                "--opus-evidence", choices=cascade.OPUS_EVIDENCE_CHOICES, default="1280"
            )
            step.add_argument(
                "--first-tier",
                choices=cascade.FIRST_TIER_CHOICES,
                default=cascade.DEFAULT_SETTINGS["first_tier"],
            )
        if name == "apply":
            step.add_argument("--wave-name", required=True)
            step.add_argument(
                "--propose", choices=proposer.SWITCH_CHOICES, default=proposer.DEFAULT_SWITCH
            )
    args = parser.parse_args()
    {
        "candidates": _run_candidates,
        "adjudicate": _run_adjudicate,
        "propose": _run_propose,
        "apply": _run_apply,
    }[args.step](args)


if __name__ == "__main__":
    main()
