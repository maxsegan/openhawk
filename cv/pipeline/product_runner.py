"""One broadcast in, 3D flights for every point out.

    python -m cv.pipeline.product_runner --video MATCH.mp4 --match-id KEY --surface clay \\
        --out $TENNIS_DATA_ROOT/processed/.../KEY [--best-of 3] [--gpu 1] [--qwen-gpus 2,3]

Steps, each resumable from its own marker under ``OUT/steps`` (rerun the same command):

1. ``upstream``: ``broadcast_runner`` with the production upstream profile (camera,
   torso-lock ball track, players, court-anchor play camera, events, automatic point
   ledger into attempts), stopped before reconstruction.
2. ``fit``: broadcast_runner's shared S6 backend on the production composed policy
   (``product_s6_policy.json`` + ``PIPELINE_COMPONENT_POLICY``), component policy through
   the shared attempt executor.
3. ``cascade``: event adjudication of the first fits' uncertain events
   (``--event-adjudication``). ``fitter``: every candidate is confirmed with provenance
   ``fitter_adjudicated``, no model is asked ($0), and the refit's physics gates decide.
   ``paid`` (rollback): the FREEZE v2 event cascade (Gemini Flash then Opus via
   OpenRouter, or local Qwen first with ``--cascade-first-tier qwen``; every paid call's
   model, tokens and dollars kept; hard cap ``--cap-usd``).
4. ``refit``: a second S6 fit of every attempt the adjudication edited, on an edited
   packet bound by its own automatic producer provenance.
5. ``assemble``: ``OUT/MATCH_RESULT.json``: per attempt the final flights (refit if
   edited, else first fit) with fitted 3D states, each tagged ``after_point_end`` by
   ``point_end_tag`` (kept, not removed), the cascade receipts and dollars, and wall,
   CPU- and GPU-occupied seconds per step.
6. ``portal``: stage the final flights for the 3D portal under ``OUT/portal/3d``.

No evaluation label is read. The event cascade refuses packets whose ball or events
are not automatic. Rollback to first fits only: ``--until fit``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

from cv.pipeline import point_end_tag, provenance
from physics import bounce_reference

REPO = Path(__file__).resolve().parents[2]
PIPELINE = Path(__file__).resolve().parent
POLICY_FILE = PIPELINE / "product_s6_policy.json"
SCHEMA = "product_match_result_v1"
DATA = Path(os.environ.get("TENNIS_DATA_ROOT", "data"))

#: The event model and operating threshold of the measured production upstream
#: (combined-v2 and cascade-mainline panels): translation-robustness ``off`` arm,
#: calibration_v3 path-marginal threshold.
EVENT_MODEL = (
    DATA / "processed/s6_local_20260911/systemic_flight_audit_v1/current_transfer_recheck_v1"
    "/upstream_event_feature_diagnosis_v1/translation_robustness_v1/training_v1/off"
    "/event_video_model.pt"
)
EVENT_MARGINAL_THRESHOLD = 0.9907168846560898
SERVE_PRIOR = DATA / "processed/serve_prior/serve_location_prior_v1.json"
#: Scoreboard reader for the point ledger (``score_vlm``); served locally only for phase A.
SCORE_VLM = "Qwen/Qwen3.8-27B-FP8"
SCORE_VLM_WEIGHTS = "models/vlm_frontier/Qwen3.8-27B-FP8"
SCORE_VLM_PORT = 8399

#: broadcast_runner flags of the production upstream. Everything else is the runner's
#: own default (torso-lock ball track and court-anchor play camera are default on).
UPSTREAM_PROFILE = (
    "--court-jobs",
    "8",
    "--court-surface-witness",
    "illumination_split",
    "--court-registration-mask",
    "court_observation",
    "--interpolated-camera-policy",
    "qualify_retry",
    "--supported-impulse-events",
    "--native-net-evidence",
    "--native-ground-evidence",
    "--event-court-geometry",
    "reliable_per_frame",
    "--event-court-frame-missing",
    "hold",
    "--event-track",
    "s4_current",
    "--live-shot-camera",
)


def _save(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, default=str) + "\n")
    temporary.replace(path)


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write_policy(out: Path) -> Path:
    """The composed policy file the shared S6 backend loads, with the prior bound."""
    policy = dict(_load(POLICY_FILE)["policy"])
    policy["serve_location_prior"] = provenance.file_record(SERVE_PRIOR)
    path = out / "s6_policy.json"
    _save(path, policy)
    return path


def upstream_command(args, python: str = sys.executable) -> list[str]:
    return [
        python,
        "-m",
        "cv.pipeline.broadcast_runner",
        "--video",
        str(args.video),
        "--out",
        str(args.out / "upstream"),
        "--match-id",
        args.match_id,
        "--surface",
        args.surface,
        "--best-of",
        str(args.best_of),
        "--deciding-tiebreak-target",
        str(args.deciding_tiebreak_target),
        "--event-model",
        str(EVENT_MODEL),
        "--event-marginal-threshold",
        repr(EVENT_MARGINAL_THRESHOLD),
        "--device",
        "0",
        *UPSTREAM_PROFILE,
        "--resume",
        "--stop-after",
        "upstream",
        *(["--max-points", str(args.max_points)] if args.max_points else []),
    ]


class GpuMeter:
    """GPU-occupied seconds of this process tree, sampled from nvidia-smi.

    A GPU counts for a sample interval when one of our descendants holds a compute
    context on it. That is occupancy, not utilisation: it is what the step reserves.
    """

    def __init__(self, interval: float = 5.0) -> None:
        self.interval = interval
        self.seconds: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _descendants(self) -> set[int]:
        try:
            table = subprocess.run(
                ["ps", "-eo", "pid=,ppid="], capture_output=True, text=True, check=True
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return set()
        children: dict[int, list[int]] = {}
        for line in table.splitlines():
            pid, ppid = (int(x) for x in line.split())
            children.setdefault(ppid, []).append(pid)
        found, stack = set(), [os.getpid()]
        while stack:
            pid = stack.pop()
            for child in children.get(pid, []):
                if child not in found:
                    found.add(child)
                    stack.append(child)
        return found

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                apps = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,gpu_uuid",
                        "--format=csv,noheader",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                ).stdout
            except (OSError, subprocess.SubprocessError):
                continue
            ours = self._descendants()
            busy = set()
            for line in apps.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) in ours:
                    busy.add(parts[1])
            for gpu in busy:
                self.seconds[gpu] = self.seconds.get(gpu, 0.0) + self.interval

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=60)

    @property
    def total(self) -> float:
        return round(sum(self.seconds.values()), 1)


def metered(step: str, out: Path, body) -> dict:
    """Run one step once: wall, children CPU and GPU-occupied seconds, then a marker."""
    marker = out / "steps" / f"{step}.json"
    if marker.is_file() and _load(marker).get("status") == "done":
        return _load(marker)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    self_before = resource.getrusage(resource.RUSAGE_SELF)
    started = time.time()
    _save(marker, {"step": step, "status": "running", "started": started, "pid": os.getpid()})
    with GpuMeter() as meter:
        detail = body()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    self_after = resource.getrusage(resource.RUSAGE_SELF)
    record = {
        "step": step,
        "status": "done",
        "started": started,
        "wall_seconds": round(time.time() - started, 1),
        "cpu_seconds": round(
            (after.ru_utime - before.ru_utime)
            + (after.ru_stime - before.ru_stime)
            + (self_after.ru_utime - self_before.ru_utime)
            + (self_after.ru_stime - self_before.ru_stime),
            1,
        ),
        "gpu_seconds": meter.total,
        "gpu_seconds_by_device": meter.seconds,
        "detail": detail or {},
    }
    _save(marker, record)
    return record


def _broadcast_runner(command: list[str], env: dict, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(f"$ {' '.join(command)}\n")
        handle.flush()
        subprocess.run(
            command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True
        )


def step_upstream(args) -> dict:
    """Phase A with the scoreboard VLM up (through the point ledger), then phase B.

    Stops before reconstruction, so a change to S6 never invalidates upstream work.
    """
    from cv.pipeline import event_cascade_models as models

    command = upstream_command(args)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "VLM_MODEL": SCORE_VLM,
        "VLM_URL": f"http://127.0.0.1:{SCORE_VLM_PORT}/v1/chat/completions",
    }
    ledger = args.out / "upstream" / args.match_id / "automatic_point_ledger_v1.csv"
    if not ledger.is_file():
        server = models.vllm_command(
            SCORE_VLM_WEIGHTS,
            SCORE_VLM,
            SCORE_VLM_PORT,
            max_model_len=32768,
            memory_fraction=0.85,
        )
        with models.vllm_servers({args.score_gpu: server}, args.out / "logs" / "score_vlm"):
            models.wait_ready((SCORE_VLM_PORT,))
            _broadcast_runner(
                [*command, "--stop-after", "point-ledger"],
                env,
                args.out / "logs" / "upstream_phase_a.log",
            )
    _broadcast_runner(command, env, args.out / "logs" / "upstream.log")
    return {"command": command, "score_vlm": SCORE_VLM}


def step_fit(args) -> dict:
    """First S6 fit of every ledger attempt: broadcast_runner's shared S6 backend."""
    from cv.pipeline.broadcast_runner import run_shared_s6_backend
    from cv.pipeline.broadcast_source import processing_source

    policy = write_policy(args.out)
    upstream = args.out / "upstream"
    match_out = upstream / args.match_id
    processing_video, _ = processing_source(
        args.video, match_out, _load(match_out / "source_integrity.json")
    )
    fit = args.out / "fit"
    if fit.exists():
        # An interrupted shared-S6 pass is kept aside; the fit starts afresh.
        fit.rename(fit.with_name(f"fit.partial.{int(time.time())}"))
    backend = argparse.Namespace(
        out=upstream,
        match_id=args.match_id,
        max_points=args.max_points,
        physical_player_motion=False,
        shared_s6_policy=policy,
        shared_s6_output=fit,
        shared_s6_workers=args.workers,
    )
    summary = _load(run_shared_s6_backend(backend, processing_video))
    return {
        "policy": provenance.file_record(policy),
        "counts": summary["counts"],
        "court_bounce_law": bounce_reference.LAW_NAME,
    }


def fit_attempts(fit: Path) -> tuple[dict, list[dict]]:
    """Every attempt of the first-fit summary, split children as their own attempts."""
    summary = _load(fit / "summary.json")
    attempts = []
    for row in summary["attempts"]:
        folder = fit / "attempts" / row["key"]
        members = (
            [(child, folder / f"child_{child['child_index']:02d}") for child in row["children"]]
            if row.get("children")
            else [(row, folder)]
        )
        for member, place in members:
            attempts.append(
                {
                    "key": member["key"],
                    "parent_key": row["key"],
                    "clip": row["clip"],
                    "status": member["status"],
                    "reason": member.get("reason"),
                    "ledger_row": row.get("ledger_row"),
                    "job": place / "job.json",
                    "stage": place / "stage",
                }
            )
    return summary, attempts


def cascade_jobs(summary: dict, attempts: list[dict], out: Path) -> list[dict]:
    """First fits in the cascade's job shape: one manifest per fitted attempt."""
    jobs = []
    for attempt in attempts:
        if attempt["status"] != "completed":
            continue
        job = _load(attempt["job"])
        manifest = out / "manifests" / f"{attempt['key']}.json"
        _save(manifest, {"policy": job["policy"], "rows": [job["row"]]})
        jobs.append(
            {
                "name": attempt["key"],
                "attempt": attempt["key"],
                "panel": summary["match_id"],
                "input_arm": "AA",
                "manifest": str(manifest),
                "stage": str(attempt["stage"]),
            }
        )
    return jobs


def _plan(job: dict) -> Path | None:
    path = Path(job["stage"]) / "component_plan.json"
    return path if path.is_file() else None


def step_cascade(args) -> dict:
    from cv.pipeline import event_cascade_models as models
    from cv.pipeline import event_cascade_runner as runner

    out = args.out / "cascade"
    summary, attempts = fit_attempts(args.out / "fit")
    jobs = cascade_jobs(summary, attempts, out)
    adjudication = getattr(args, "event_adjudication", None) or runner.cascade.DEFAULT_ADJUDICATION
    scope = runner.cascade.candidate_scope_for(adjudication)
    candidates = runner.build_candidates(jobs, scope=scope, plan_for=_plan)
    candidates["wave_attempts"] = len(attempts)
    _save(out / "candidates.json", candidates)
    if adjudication == "fitter":
        document, _cost = runner.fitter_adjudicate(out, attempts=len(attempts))
        wave = runner.build_wave(jobs, document["decisions"], out, f"{args.match_id}_cascade")
        _save(out / "CASCADE_WAVE.json", wave)
        return {
            "adjudication": adjudication,
            "candidate_scope": scope,
            "candidates": candidates["candidates"],
            "decisions": document["actions"],
            "edited_attempts": len(wave["jobs"]),
            "receipt_usd": document["receipt_usd"],
        }
    ports = tuple(8523 + n for n in range(len(args.qwen_gpus)))
    first_tier = (
        getattr(args, "cascade_first_tier", None) or runner.cascade.DEFAULT_SETTINGS["first_tier"]
    )
    chosen = runner.cascade.settings({"first_tier": first_tier})
    if candidates["rows"]:
        local = first_tier == "qwen" and any(
            not models.reply_is_good(models.reply_path(out / "adjudicate", "qwen", 1, row, chosen))
            for row in candidates["rows"]
        )
        servers = (
            models.qwen_servers(args.qwen_gpus, ports, args.out / "logs" / "vllm")
            if local
            else contextlib.nullcontext()
        )
        with servers:
            document, cost = runner.adjudicate(
                out,
                qwen_ports=",".join(map(str, ports)),
                cap_usd=args.cap_usd,
                first_tier=first_tier,
            )
    else:
        document = runner.decision_document([], {}, {}, chosen)
        _save(out / "adjudicate" / "decisions.json", document)
        cost = runner.cost_summary(
            document, attempts=len(attempts), qwen_gpu_seconds=0.0, qwen_new_calls=0
        )
        _save(out / "COST.json", cost)
    wave = runner.build_wave(jobs, document["decisions"], out, f"{args.match_id}_cascade")
    _save(out / "CASCADE_WAVE.json", wave)
    return {
        "adjudication": adjudication,
        "candidate_scope": scope,
        "candidates": candidates["candidates"],
        "decisions": document["actions"],
        "edited_attempts": len(wave["jobs"]),
        "receipt_usd": document["receipt_usd"],
    }


def step_refit(args) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    from cv.pipeline.s6_broadcast_backend import run_stage

    wave = _load(args.out / "cascade" / "CASCADE_WAVE.json")
    root = args.out / "refit"

    def one(job: dict) -> dict:
        manifest = _load(Path(job["manifest"]))
        folder = root / job["attempt"]
        result = _load(folder / "run.json") if (folder / "run.json").is_file() else None
        if result is None:
            folder.mkdir(parents=True, exist_ok=True)
            _save(folder / "job.json", {"row": manifest["rows"][0], "policy": manifest["policy"]})
            result = run_stage(folder / "job.json", folder / "stage", job["timeout_seconds"])
            _save(folder / "run.json", result)
        return {"attempt": job["attempt"], **result}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        runs = list(pool.map(one, wave["jobs"]))
    _save(root / "REFIT.json", {"runs": runs})
    return {
        "refits": len(runs),
        "completed": sum(r["status"] == "completed" for r in runs),
        "court_bounce_law": bounce_reference.LAW_NAME,
    }


def dense_flights(result: dict) -> dict[tuple, dict]:
    """Fitted 3D flights of one S6 result keyed by (component index, flight index)."""
    found = {}
    for component in result.get("component_results") or []:
        measurement = (component.get("result") or {}).get("measurement") or {}
        for index, flight in enumerate(measurement.get("dense_flights") or []):
            found[(component["component_index"], index)] = flight
    for index, flight in enumerate((result.get("measurement") or {}).get("dense_flights") or []):
        found[(None, index)] = flight
    return found


def _flight_record(verdict_flight: dict, dense: dict | None, step: int) -> dict:
    record = {
        "role": verdict_flight.get("role"),
        "start_frame": verdict_flight.get("start_frame"),
        "end_frame": verdict_flight.get("end_frame"),
        "original_flight_index": verdict_flight.get("original_flight_index"),
    }
    if dense is not None:
        record.update(
            start_xyz_m=dense["start_xyz"],
            end_xyz_m=dense["end_xyz"],
            bounces_xyz_m=[bounce["x"] for bounce in dense.get("bounces") or []],
            net_hits=len(dense.get("net_hits") or []),
            positions_m=dense["positions"][::step],
            positions_note=f"dense 240 Hz samples, every {step}th kept",
        )
    return record


def attempt_flights(result: dict, step: int = 4) -> list[dict]:
    """Accepted flights of one S6 result with their fitted 3D states."""
    dense = dense_flights(result)
    flights = []
    for flight in (result.get("verdict") or {}).get("flights") or []:
        if not flight.get("accepted"):
            continue
        key = (flight.get("component_index"), flight.get("flight_index"))
        match = dense.get(key)
        if (
            match is not None
            and abs(float(match["start_frame"]) - float(flight["start_frame"])) > 1
        ):
            match = None
        flights.append(_flight_record(flight, match, step))
    return flights


def step_assemble(args) -> dict:
    summary, attempts = fit_attempts(args.out / "fit")
    refits = {}
    if (args.out / "refit" / "REFIT.json").is_file():
        refits = {r["attempt"]: r for r in _load(args.out / "refit" / "REFIT.json")["runs"]}
    decisions = _load(args.out / "cascade" / "adjudicate" / "decisions.json")
    cost = _load(args.out / "cascade" / "COST.json")
    rows = []
    for attempt in attempts:
        refit = refits.get(attempt["key"])
        if refit and refit["status"] == "completed":
            stage, source = args.out / "refit" / attempt["key"] / "stage", "cascade_refit"
        elif attempt["status"] == "completed":
            stage, source = attempt["stage"], "first_fit"
        else:
            stage, source = None, None
        flights, point_end = [], None
        if stage:
            result = _load(stage / "result.json")
            flights = attempt_flights(result)
            # Post-processing only: flights after an automatic point-end call stay
            # emitted, marked ``after_point_end`` for consumers to drop or keep.
            point_end = point_end_tag.tag_result(result, flights)
        ledger = attempt.get("ledger_row") or {}
        rows.append(
            {
                "key": attempt["key"],
                "clip": attempt["clip"],
                "ledger": ledger,
                "first_fit_status": attempt["status"],
                "held_reason": attempt.get("reason"),
                "cascade_edited": refit is not None,
                "cascade_refit_status": refit["status"] if refit else None,
                "final_source": source,
                "result": provenance.file_record(stage / "result.json") if stage else None,
                "accepted_flights": flights,
                "point_end_call": point_end,
            }
        )
    steps = {step: _load(args.out / "steps" / f"{step}.json") for step in STEPS[:-2]}
    document = {
        "schema": SCHEMA,
        "match_id": args.match_id,
        "source_video": provenance.file_record(args.video),
        "code": provenance.git_record(REPO),
        "observation_origin": "automatic",
        "evaluation_labels_read": False,
        # The law the fitting steps ran under; a step marker without one predates the switch.
        "court_bounce_law": {
            **bounce_reference.law_record(),
            "fit_steps": {
                step: (steps[step].get("detail") or {}).get("court_bounce_law")
                for step in ("fit", "refit")
            },
        },
        "counts": {
            "ledger_attempts": len(summary["attempts"]),
            "attempts_including_split_children": len(attempts),
            "first_fit_completed": sum(a["status"] == "completed" for a in attempts),
            "held_or_failed_before_fit": sum(a["status"] != "completed" for a in attempts),
            "attempts_with_flights": sum(bool(r["accepted_flights"]) for r in rows),
            "accepted_flights": sum(len(r["accepted_flights"]) for r in rows),
            "accepted_flights_after_point_end": sum(
                f["after_point_end"] for r in rows for f in r["accepted_flights"]
            ),
            "event_adjudication": (decisions.get("settings") or {}).get("adjudication", "paid"),
            "cascade_candidates": decisions["candidates"],
            "cascade_actions": decisions["actions"],
            "cascade_edited_attempts": len(refits),
        },
        "cost": {
            "paid_usd": decisions["receipt_usd"],
            "by_model": decisions["by_model"],
            "paid_calls": sum(
                m["calls"] for k, m in decisions["by_model"].items() if "qwen" not in k.lower()
            ),
            "note": "Dollars actually spent on this match; each receipt is in cascade/adjudicate.",
        },
        "time": {
            "wall_seconds": round(sum(s["wall_seconds"] for s in steps.values()), 1),
            "cpu_seconds": round(sum(s["cpu_seconds"] for s in steps.values()), 1),
            "gpu_seconds": round(sum(s["gpu_seconds"] for s in steps.values()), 1),
            "by_step": {
                name: {k: s[k] for k in ("wall_seconds", "cpu_seconds", "gpu_seconds")}
                for name, s in steps.items()
            },
            "qwen_gpu_seconds_in_calls": cost.get("qwen_gpu_seconds_new"),
        },
        "yield": None,
        "yield_note": "No labels exist for this broadcast; no reference-flight yield is claimed.",
        "attempts": rows,
    }
    _save(args.out / "MATCH_RESULT.json", document)
    return document["counts"]


def step_portal(args) -> dict:
    """Stage the final flights for the 3D portal (never the live portal directly).

    First fits go through the automatic exporter; each cascade refit then replaces its
    attempt's entry, exported from the edited packet and its own producer provenance.
    """
    from cv.viz import export_automatic_s6
    from cv.viz import export_connected_3d as legacy
    from cv.viz import export_local_s6 as shared
    from scripts.shared_data import repository_root, resolve_shared_root

    staging = args.out / "portal" / "3d"
    label = f"Product runner · {args.match_id}"
    report = export_automatic_s6.export_run(
        args.out / "fit", staging, merge=staging.exists(), run_label=f"{label} · first fit"
    )
    repo = repository_root()
    root = resolve_shared_root(repo, None)
    wave = _load(args.out / "cascade" / "CASCADE_WAVE.json")
    adjudicated = "fitter adjudication" if wave.get("adjudication") == "fitter" else "event cascade"
    replaced, failed = [], []
    index_path = staging / "data" / "index.json"
    index = legacy._read_json(index_path)
    for job in wave["jobs"]:
        stage = args.out / "refit" / job["attempt"] / "stage"
        if not (stage / "result.json").is_file():
            continue
        manifest_path = Path(job["manifest"])
        row = _load(manifest_path)["rows"][0]
        try:
            entry = shared.export_case(
                row,
                stage / "result.json",
                manifest_path,
                staging,
                root,
                repo,
                f"{label} · after {adjudicated}",
            )
        except Exception as error:  # noqa: BLE001 - recorded; the first fit stays shown
            failed.append({"attempt": job["attempt"], "error": f"{type(error).__name__}: {error}"})
            continue
        entry["product_final_source"] = "cascade_refit"
        index["points"] = [p for p in index["points"] if p.get("key") != entry.get("key")] + [entry]
        replaced.append(job["attempt"])
    index["count"] = len(index["points"])
    legacy._write_json(index_path, index, pretty=True)
    return {
        "staging": str(staging),
        "exported_attempts": report["exported_attempts"],
        "held_before_fitting": len(report["held_before_fitting"]),
        "cascade_refits_replaced": replaced,
        "cascade_refit_export_failures": failed,
    }


STEPS = ("upstream", "fit", "cascade", "refit", "assemble", "portal")


def run(args) -> dict:
    args.out.mkdir(parents=True, exist_ok=True)
    records = {}
    for step in STEPS:
        records[step] = metered(step, args.out, lambda step=step: globals()[f"step_{step}"](args))
        if step == args.until:
            break
    return records


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--match-id", required=True)
    p.add_argument("--surface", choices=("hard", "clay", "grass"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--best-of", type=int, choices=(3, 5), default=3)
    p.add_argument("--deciding-tiebreak-target", type=int, choices=(7, 10), default=7)
    p.add_argument("--gpu", type=int, default=0, help="physical GPU for the upstream models")
    p.add_argument(
        "--score-gpu", type=int, default=3, help="physical GPU for the scoreboard VLM (phase A)"
    )
    p.add_argument(
        "--max-points", type=int, help="fit only the first N ledger attempts (smoke runs)"
    )
    p.add_argument("--workers", type=int, default=6, help="parallel S6 attempt processes")
    p.add_argument(
        "--qwen-gpus",
        type=lambda text: tuple(int(x) for x in text.split(",") if x.strip()),
        default=(2, 3),
        help="physical GPUs for the local Qwen servers of the event cascade",
    )
    p.add_argument("--cap-usd", type=float, default=10.0, help="hard cap on paid model calls")
    p.add_argument(
        "--event-adjudication",
        choices=("fitter", "paid"),
        default=None,
        help=(
            "fitter: confirm every uncertain event, no model, the refit decides ($0); "
            "paid: the model cascade (rollback). Default: event_cascade.DEFAULT_ADJUDICATION"
        ),
    )
    p.add_argument(
        "--cascade-first-tier",
        choices=("qwen", "gemini"),
        default=None,
        help=(
            "paid cascade first tier: local Qwen (FREEZE v2) or Gemini Flash (no Qwen servers). "
            "Default: event_cascade.DEFAULT_SETTINGS"
        ),
    )
    p.add_argument("--until", choices=STEPS, help="stop after this step")
    return p


def main() -> None:
    args = parser().parse_args()
    args.video = args.video.resolve()
    args.out = args.out.resolve()
    print(json.dumps(run(args), indent=2, default=str))


if __name__ == "__main__":
    main()
