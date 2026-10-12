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
5. ``beam``: the event beam (``event_beam``; ``--event-beam off`` rolls back): a per-match
   audio delay calibration, then label-free contact candidates from independent time
   witnesses re-fitted per attempt on top of the cascade's decisions; an attempt's final
   fit moves to a beam fit only when it accepts more flights and loses none.
6. ``assemble``: ``OUT/MATCH_RESULT.json``: per attempt the final flights (beam if it
   moved, else refit if edited, else first fit) with fitted 3D states, each tagged
   ``after_point_end`` by ``point_end_tag`` and with its post-validation
   (``flight_validation``; ``--flight-validation off`` rolls back): flights failing it stay
   listed with their reasons but are not ``confirmed``. Also the audio delay calibration,
   the cascade receipts and dollars, and wall, CPU- and GPU-occupied seconds per step.
7. ``portal``: stage the final flights for the 3D portal under ``OUT/portal/3d``.
8. ``pose``: 3D player pose (near and far, 17 COCO joints, every native frame of every staged
   attempt window; ``cv.pipeline.player_pose_3d``) into ``OUT/pose`` and per-clip portal pose
   layers bound into the documents (``TENNIS_PLAYER_POSE_3D=off``: rollback, skipped).
9. ``compact``: first the overlay-video layer (court, player boxes, 2D skeletons, fitted
   flights) is written into the attempt documents from the run's own fitter cameras and pose
   (``cv.pipeline.portal_flight_overlay``; ``TENNIS_PORTAL_FLIGHT_OVERLAY=off``: rollback);
   then bounded per-match disk (``product_artifacts``): portal pictures become one
   H.264 clip per kept attempt, S6 stage trees and upstream outputs go into verified
   archives, event crops and native frame folders are deleted (``--keep full``: nothing).

``--score-vlm-url`` / ``--qwen-ports`` use queue-owned servers of the same model
(``cv.pipeline.product_queue``) instead of starting servers per match.

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

from cv.pipeline import (
    best_effort_tier,
    failure_index,
    ffmpeg_pin,
    fit_only_bounce_rules,
    flight_validation,
    gpu_lease,
    missed_play,
    nvdec_video,
    point_end_tag,
    pose_gpu,
    post_fit_checks,
    product_outcomes,
    provenance,
    switch_fallbacks,
)
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
    # Shade-tolerant line mask after every other route fails (court-gaps 2026-10-02: +8
    # sun/shadow attempts on Muguruza-Halep, 0 of 840 displaced courts); wide-view retry for low
    # main cameras (court-paris 2026-10-03: De Minaur-Kecmanovic 0 -> 193/200 courts; held-out
    # 54 wide courts on 3 matches, 0 wrong). Rollback: illumination_split_shade_lines.
    "illumination_split_shade_lines_wide_view",
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
    # Owner 2026-09-29: heal before refusing. Regular 25->50 / 25->30 repeats are restored to
    # native cadence; what cannot be healed is processed point-gated (broadcast_source).
    "--cadence-restore",
)


def _save(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, default=str) + "\n")
    temporary.replace(path)


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def write_policy(out: Path) -> Path:
    """The composed policy file the shared S6 backend loads, with the prior bound.

    ``PIPELINE_COMPONENT_POLICY`` is written out explicitly (same effective policy), so a
    default changed in the code while a long queue runs cannot reinterpret this match's file
    between its fit and its portal export.
    """
    from cv.pipeline.s6_labeled_stage import pipeline_policy

    policy = pipeline_policy(dict(_load(POLICY_FILE)["policy"]))
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
        *(["--concurrent-tracking"] if getattr(args, "concurrent_tracking", False) else []),
        *(["--long-shot-view-windows"] if getattr(args, "long_shot_view_windows", False) else []),
        *(
            ["--scoreboard-map-agreement"]
            if getattr(args, "scoreboard_map_agreement", False)
            else []
        ),
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
        #: Device utilisation (percent x seconds) while we held it; includes any other
        #: tenant of the same GPU, so it is an upper bound on ours.
        self.utilisation: dict[str, float] = {}
        #: Peak MiB our processes held on each device.
        self.peak_mib: dict[str, float] = {}
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
                        "--query-compute-apps=pid,gpu_uuid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                ).stdout
                devices = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=uuid,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                ).stdout
            except (OSError, subprocess.SubprocessError):
                continue
            ours = self._descendants()
            busy: dict[str, float] = {}
            for line in apps.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) == 3 and parts[0].isdigit() and int(parts[0]) in ours:
                    memory = float(parts[2]) if parts[2].replace(".", "").isdigit() else 0.0
                    busy[parts[1]] = busy.get(parts[1], 0.0) + memory
            load = {}
            for line in devices.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) == 2 and parts[1].isdigit():
                    load[parts[0]] = float(parts[1])
            for gpu, memory in busy.items():
                self.seconds[gpu] = self.seconds.get(gpu, 0.0) + self.interval
                self.utilisation[gpu] = (
                    self.utilisation.get(gpu, 0.0) + load.get(gpu, 0.0) * self.interval
                )
                self.peak_mib[gpu] = max(self.peak_mib.get(gpu, 0.0), memory)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=60)

    @property
    def total(self) -> float:
        return round(sum(self.seconds.values()), 1)

    @property
    def mean_utilisation(self) -> dict[str, float]:
        """Mean device utilisation (percent) over the seconds we held each device."""
        return {
            gpu: round(self.utilisation.get(gpu, 0.0) / seconds, 1)
            for gpu, seconds in self.seconds.items()
            if seconds
        }


def _runner_alive(pid: int) -> bool:
    """A live product runner (or reprocess) process: its pid exists and runs our module."""
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return False
    return (
        "product_runner" in command or "product_reprocess" in command or "product_queue" in command
    )


def metered(step: str, out: Path, body) -> dict:
    """Run one step once: wall, children CPU and GPU-occupied seconds, then a marker."""
    marker = out / "steps" / f"{step}.json"
    if marker.is_file() and _load(marker).get("status") == "done":
        return _load(marker)
    if marker.is_file():
        owner = _load(marker).get("pid")
        if _load(marker).get("status") == "running" and owner and owner != os.getpid():
            if _runner_alive(owner):
                raise RuntimeError(f"step {step} is already running in pid {owner}")
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    self_before = resource.getrusage(resource.RUSAGE_SELF)
    started = time.time()
    # The switches the step runs under (cv.pipeline.switch_fallbacks: reruns replay them).
    switches = switch_fallbacks.snapshot()
    _save(
        marker,
        {
            "step": step,
            "status": "running",
            "started": started,
            "pid": os.getpid(),
            "switches": switches,
        },
    )
    with GpuMeter() as meter, step_fallback_log(step, out) as fallbacks:
        try:
            detail = body()
        except BaseException as error:
            _save(
                marker,
                {
                    "step": step,
                    "status": "failed",
                    "started": started,
                    "wall_seconds": round(time.time() - started, 1),
                    "error": f"{type(error).__name__}: {error}"[:2000],
                    "fallbacks": fallbacks(),
                    "switches": switches,
                },
            )
            raise
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
        "gpu_utilisation_percent_by_device": meter.mean_utilisation,
        "gpu_peak_mib_by_device": meter.peak_mib,
        "detail": detail or {},
        "ffmpeg": ffmpeg_pin.identity(),
        "fallbacks": fallbacks(),
        "switches": switches,
        # The match folder after the step; bulk only grows until compact, so the largest
        # value over the steps is the match's local disk high-water (RECEIPT.json).
        "disk_bytes_after": _disk_bytes(out),
    }
    _save(marker, record)
    return record


def _disk_bytes(out: Path) -> int | None:
    from cv.pipeline.product_artifacts import disk_bytes

    try:
        return disk_bytes(out)
    except OSError:
        return None


#: Steps whose solves use the GPU fit (a box that declined it fits them on the CPU).
GPU_FIT_STEPS = ("fit", "refit", "beam", "cascade")


@contextlib.contextmanager
def step_fallback_log(step: str, out: Path):
    """Point every process of the step at ``logs/fallbacks_<step>.jsonl``
    (``cv.pipeline.switch_fallbacks``); yields the summary function for the step marker.

    A previous attempt's log is kept beside it, timestamped. A box that declined the GPU fit
    (``TENNIS_S6_GPU_FIT_DECLINED``, set by product_boxes) is recorded for each fit step.
    """
    log = out / "logs" / f"fallbacks_{step}.jsonl"
    if log.is_file():
        log.rename(log.with_name(f"fallbacks_{step}.{int(log.stat().st_mtime)}.jsonl"))
    previous = os.environ.get(switch_fallbacks.ENV)
    os.environ[switch_fallbacks.ENV] = str(log)
    switch_fallbacks.reset()
    declined = os.environ.get(switch_fallbacks.GPU_FIT_DECLINED_ENV)
    if declined and step in GPU_FIT_STEPS:
        switch_fallbacks.record("TENNIS_S6_GPU_FIT", "box_declined", declined, where=step)
    try:
        yield lambda: switch_fallbacks.summary(log)
    finally:
        if previous is None:
            os.environ.pop(switch_fallbacks.ENV, None)
        else:
            os.environ[switch_fallbacks.ENV] = previous


def _last_error(log: Path) -> str | None:
    """The last exception line of a log (``ValueError: source cadence rejected: ...``)."""
    try:
        lines = log.read_text(errors="replace").splitlines()[-400:]
    except OSError:
        return None
    for line in reversed(lines):
        text = line.strip()
        head = text.split(":", 1)[0]
        if ":" in text and head.split(".")[-1].endswith(("Error", "Exception", "Failure")):
            return text[:1000]
    return None


def _broadcast_runner(command: list[str], env: dict, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(f"$ {' '.join(command)}\n")
        handle.flush()
        code = subprocess.run(
            command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False
        ).returncode
    if code:
        # The step marker (and OUTCOMES.json) carry the cause, not only the exit status.
        cause = _last_error(log) or "no exception line in the log"
        raise RuntimeError(f"broadcast_runner exited {code}: {cause} (log {log.name})")


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
        "VLM_URL": args.score_vlm_url or f"http://127.0.0.1:{SCORE_VLM_PORT}/v1/chat/completions",
    }
    if nvdec_video.requested_gpu() is not None:
        # NVDEC on the leased card: device 0 under CUDA_VISIBLE_DEVICES above. An index from
        # the dispatcher's environment would name another card (0 = the scoreboard card).
        env["TENNIS_VIDEO_DECODE"], env["TENNIS_NVDEC_GPU"] = "nvdec", "0"
    if getattr(args, "gpu_lease", None):
        env[gpu_lease.ENV_SPEC] = args.gpu_lease
        env[gpu_lease.ENV_LOG] = str(args.out / "logs" / "gpu_leases.jsonl")
        env.pop(gpu_lease.ENV_HELD, None)
    ledger = args.out / "upstream" / args.match_id / "automatic_point_ledger_v1.csv"
    if not ledger.is_file() and args.score_vlm_url:
        # A queue-owned server of the same model (cv.pipeline.product_queue).
        _broadcast_runner(
            [*command, "--stop-after", "point-ledger"],
            env,
            args.out / "logs" / "upstream_phase_a.log",
        )
    elif not ledger.is_file():
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
    return {
        "command": command,
        "score_vlm": SCORE_VLM,
        "score_vlm_server": "shared" if args.score_vlm_url else "per_match",
    }


def step_fit(args) -> dict:
    """First S6 fit of every ledger attempt: broadcast_runner's shared S6 backend."""
    from cv.pipeline.broadcast_runner import run_shared_s6_backend
    from cv.pipeline.s6_broadcast_backend import gpu_fit_route
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
        video=args.video,
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
        "gpu_fit_route": gpu_fit_route(),
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
    shared = bool(args.qwen_ports)
    ports = args.qwen_ports or tuple(8523 + n for n in range(len(args.qwen_gpus)))
    first_tier = (
        getattr(args, "cascade_first_tier", None) or runner.cascade.DEFAULT_SETTINGS["first_tier"]
    )
    if first_tier == "none":
        raise SystemExit("--cascade-first-tier none: paid adjudication needs a model first tier")
    chosen = runner.cascade.settings({"first_tier": first_tier})
    if candidates["rows"]:
        local = (
            not shared
            and first_tier == "qwen"
            and any(
                not models.reply_is_good(
                    models.reply_path(out / "adjudicate", "qwen", 1, row, chosen)
                )
                for row in candidates["rows"]
            )
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
        "qwen_server": "shared" if shared else "per_match",
    }


def in_match(recorded: str | Path, out: Path) -> Path:
    """A file path a step recorded (``CASCADE_WAVE.json`` jobs, ``BEAM.json``), read from this
    copy of the match: steps record absolute paths, and the match may since have moved (stage
    placement runs a step on another box at the same data-relative path; a published or
    restored match lives elsewhere). The moved-match rule of
    ``resolution.declared_source_path``: the file under this match directory first, the
    recorded path as the fallback (unchanged for a match read where it was made)."""
    from cv.pipeline import resolution

    return resolution.declared_source_path(recorded, out / "MATCH_RESULT.json")


def _refit_one(job: dict, root: Path) -> dict:
    """One cascade refit (module level: the attempt pool may run it in a worker process)."""
    from cv.pipeline.s6_broadcast_backend import run_stage

    manifest = _load(in_match(job["manifest"], root.parent))
    folder = root / job["attempt"]
    result = _load(folder / "run.json") if (folder / "run.json").is_file() else None
    if result is None:
        folder.mkdir(parents=True, exist_ok=True)
        # An interrupted refit left a stage without its run record: kept aside, redone
        # (run_stage refuses an existing output and would record a false failure).
        for partial in (folder / "stage", folder / "stage_run"):
            if partial.is_symlink():
                partial.unlink()
            elif partial.exists():
                partial.rename(folder / f"{partial.name}.partial.{int(time.time())}")
        _save(folder / "job.json", {"row": manifest["rows"][0], "policy": manifest["policy"]})
        result = run_stage(folder / "job.json", folder / "stage", job["timeout_seconds"])
        _save(folder / "run.json", result)
    return {"attempt": job["attempt"], **result}


def step_refit(args) -> dict:
    from cv.pipeline.s6_broadcast_backend import (
        attempt_pool,
        gpu_fit_error_log,
        gpu_fit_on,
        gpu_fit_route,
    )

    wave = _load(args.out / "cascade" / "CASCADE_WAVE.json")
    root = args.out / "refit"
    root.mkdir(parents=True, exist_ok=True)
    with gpu_fit_error_log(root) as fallbacks, attempt_pool(args.workers) as pool:
        runs = list(pool.map(_refit_one, wave["jobs"], [root] * len(wave["jobs"])))
    _save(root / "REFIT.json", {"runs": runs})
    return {
        "refits": len(runs),
        "completed": sum(r["status"] == "completed" for r in runs),
        "court_bounce_law": bounce_reference.LAW_NAME,
        **({"gpu_fit_cpu_fallbacks": fallbacks()} if gpu_fit_on() else {}),
        "gpu_fit_route": gpu_fit_route(),
    }


def _cascade_manifest(args, key: str) -> Path:
    return args.out / "cascade" / "manifests" / f"{key}.json"


def step_beam(args) -> dict:
    """Event beam over every completed attempt (see ``event_beam``)."""

    from cv.pipeline import event_beam
    from cv.pipeline import event_cascade_rebind as rebind
    from cv.pipeline.s6_broadcast_backend import (
        attempt_pool,
        gpu_fit_error_log,
        gpu_fit_on,
        gpu_fit_route,
    )

    if getattr(args, "event_beam", "on") == "off":
        return {"event_beam": "off"}
    _summary, attempts = fit_attempts(args.out / "fit")
    wave = _load(args.out / "cascade" / "CASCADE_WAVE.json")
    timeouts = {j["attempt"]: j.get("timeout_seconds", 4200) for j in wave["jobs"]}
    decisions = _load(args.out / "cascade" / "adjudicate" / "decisions.json")["decisions"]
    refits = {}
    if (args.out / "refit" / "REFIT.json").is_file():
        refits = {r["attempt"]: r for r in _load(args.out / "refit" / "REFIT.json")["runs"]}
    beam_attempts = []
    for attempt in attempts:
        manifest = _cascade_manifest(args, attempt["key"])
        if attempt["status"] != "completed" or not manifest.is_file():
            continue
        refit = refits.get(attempt["key"])
        baseline = (
            args.out / "refit" / attempt["key"] / "stage" / "result.json"
            if refit and refit["status"] == "completed"
            else attempt["stage"] / "result.json"
        )
        settled = rebind.settling([d for d in decisions if d.get("attempt") == attempt["key"]])
        beam_attempts.append(
            event_beam.Attempt.build(
                attempt["key"],
                _load(manifest),
                settled,
                baseline,
                timeouts.get(attempt["key"], 4200),
            )
        )
    match = event_beam.load_match(args.out / "upstream", args.match_id)

    (args.out / "beam").mkdir(parents=True, exist_ok=True)
    with gpu_fit_error_log(args.out / "beam") as fallbacks, attempt_pool(args.workers) as pool:

        def fit_many(jobs):
            list(pool.map(event_beam.fit_one, *zip(*jobs))) if jobs else None

        summary = event_beam.run(
            match,
            beam_attempts,
            args.out / "beam",
            fit_many,
            prune=getattr(args, "event_beam_prune", "off"),
            # A reprocess of some attempts reuses the match's own calibration (product_reprocess).
            calibration=getattr(args, "beam_calibration", None),
        )
    return {k: summary[k] for k in ("attempts", "candidates", "fits", "attempts_changed")} | {
        "audio_delay_frames": summary["calibration"]["delay_frames"],
        "audio_delay_confident": summary["calibration"]["confident"],
        **({"gpu_fit_cpu_fallbacks": fallbacks()} if gpu_fit_on() else {}),
        "gpu_fit_route": gpu_fit_route(),
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


def validate_flights(result: dict, row: dict, flights: list[dict]) -> dict:
    """Post-validation tags on ``flights`` (``attempt_flights`` records, by start frame);
    a flight failing it gets ``confirmed`` False. Returns the reasons tally."""
    from cv.pipeline import event_beam
    from cv.pipeline import flight_validation as fv

    cameras = _load(event_beam._resolve(row["cameras"]))
    boxes = event_beam.player_boxes(row)
    rows = fv.validate(result, lambda f, xyz: fv.foot_reach(cameras, boxes, f, xyz))
    by_start = {round(float(r["start_frame"]), 3): r for r in rows}
    dense = fv._dense(result)
    tally: dict[str, int] = {}
    for flight in flights:
        r = by_start.get(round(float(flight["start_frame"]), 3))
        if r is None:
            continue
        flight["validation"] = {k: r[k] for k in ("ok", "reasons", "tags")}
        flight["confirmed"] = bool(r["ok"])
        for reason in r["reasons"]:
            tally[reason] = tally.get(reason, 0) + 1
        if r.get("trim"):
            _trim_record(flight, r["trim"], dense.get(round(float(flight["start_frame"]))))
            key = f"trim:{r['trim']['reason']}"
            tally[key] = tally.get(key, 0) + 1
    return tally


def _trim_record(flight: dict, trim: dict, dense: dict | None, step: int = 4) -> None:
    """Cut an ``attempt_flights`` record at ``trim["end_frame"]`` (TENNIS_FLIGHT_TRIM); the
    original end and the reason stay in ``trim``."""
    if dense is None or not flight.get("positions_m"):
        return
    start, end = float(dense["start_frame"]), float(dense["end_frame"])
    cut = float(trim["end_frame"])
    n = len(dense["positions"])
    keep = int((cut - start) / (end - start) * (n - 1)) if end > start else 0
    positions = dense["positions"][: keep + 1]
    flight["trim"] = {**trim, "end_frame_before": flight["end_frame"]}
    flight["end_frame"] = min(float(flight["end_frame"]), cut)
    flight["end_xyz_m"] = positions[-1]
    flight["positions_m"] = positions[::step]
    flight["bounces_xyz_m"] = [
        b["x"] for b in dense.get("bounces") or [] if float(b["frame"]) <= cut and "x" in b
    ]
    flight["net_hits"] = sum(float(h["frame"]) <= cut for h in dense.get("net_hits") or [])


def _cut_continuation(flight: dict, dense: dict | None, end: float, step: int = 4) -> None:
    """Fit-only bounce rule 1: drop the fitted states past the verdict flight's end (a modelled
    continuation, never validated) from an ``attempt_flights`` record; recorded in
    ``continuation``. A trim that already cut earlier wins."""
    if dense is None or not flight.get("positions_m"):
        return
    start, dense_end = float(dense["start_frame"]), float(dense["end_frame"])
    after = [b for b in dense.get("bounces") or [] if float(b["frame"]) >= end]
    if dense_end <= end + fit_only_bounce_rules.CONTINUATION_EPS and not after:
        return
    flight["continuation"] = {
        "label": fit_only_bounce_rules.CONTINUATION,
        "from_frame": end,
        "dense_end_frame": dense_end,
        "after_bounces": len(after),
        "rules_version": fit_only_bounce_rules.RULES_VERSION,
    }
    if flight.get("trim") and float(flight["end_frame"]) <= end:
        return
    n = len(dense["positions"])
    keep = int((end - start) / (dense_end - start) * (n - 1)) if dense_end > start else 0
    positions = dense["positions"][: keep + 1]
    flight["end_xyz_m"] = positions[-1]
    flight["positions_m"] = positions[::step]
    flight["bounces_xyz_m"] = [
        b["x"] for b in dense.get("bounces") or [] if float(b["frame"]) < end and "x" in b
    ]
    flight["net_hits"] = sum(float(h["frame"]) <= end for h in dense.get("net_hits") or [])


def apply_fit_only_bounce_rules(
    result: dict,
    flights: list[dict],
    detector: fit_only_bounce_rules.DetectorRecord,
    clip: str,
    packet: Path | None,
) -> dict[str, int]:
    """``TENNIS_FIT_ONLY_BOUNCE_RULES=on``: rules 1-3 on one attempt's ``attempt_flights``
    records (after ``validate_flights``). A demoted flight gets ``confirmed`` False, its reason
    in ``validation.reasons`` and the evidence in ``fit_only_bounce_rules``; every record is cut
    at its verdict end. Returns the tally."""
    events = (
        fit_only_bounce_rules.packet_events(_load(packet))
        if packet is not None and packet.is_file()
        else None
    )

    def verdict_span(f: dict) -> tuple[float, float]:
        end = (f.get("trim") or {}).get("end_frame_before", f["end_frame"])
        return fit_only_bounce_rules.span_key({"start_frame": f["start_frame"], "end_frame": end})

    spans = [verdict_span(f) for f in flights if f.get("confirmed")]
    decisions = fit_only_bounce_rules.evaluate_result(result, events, detector, clip, spans)
    dense = flight_validation._dense(result)
    tally: dict[str, int] = {}
    for flight in flights:
        span = verdict_span(flight)
        _cut_continuation(flight, dense.get(round(float(flight["start_frame"]))), span[1])
        if flight.get("continuation"):
            tally["continuation_cut"] = tally.get("continuation_cut", 0) + 1
        decision = decisions.get(span) if flight.get("confirmed") else None
        if decision is None:
            continue
        fit_only = [b for b in decision["bounces"] if b["fit_only"]]
        if not fit_only:
            continue
        flight["fit_only_bounce_rules"] = decision
        tally["flights_with_fit_only_bounce"] = tally.get("flights_with_fit_only_bounce", 0) + 1
        if decision["demote"]:
            flight["confirmed"] = False
            validation = flight.setdefault("validation", {"ok": False, "reasons": [], "tags": []})
            validation["ok"] = False
            validation["reasons"] = list(validation.get("reasons") or []) + decision["reasons"]
            key = f"demoted:{decision['reason']}"
            tally[key] = tally.get(key, 0) + 1
        else:
            tally["kept_witnessed_or_after_only"] = tally.get("kept_witnessed_or_after_only", 0) + 1
    return tally


def _demoted(flight: dict) -> bool:
    """Left the validated tier by the fit-only bounce rules or the post-fit checks."""
    return bool(
        (flight.get("fit_only_bounce_rules") or {}).get("demote")
        or (flight.get("post_fit_checks") or {}).get("demote")
    )


def _demotion_reasons(flight: dict) -> list[str]:
    out = []
    for block in ("fit_only_bounce_rules", "post_fit_checks"):
        decision = flight.get(block) or {}
        if decision.get("demote"):
            out += list(decision["reasons"])
    return out


def _trim_measures(measures: dict, keep: int = 5) -> dict:
    """Per-frame residual lists cut to the ``keep`` largest (the record stays small)."""
    out = {}
    for key, value in measures.items():
        if isinstance(value, list) and value and isinstance(value[0], list):
            out[key] = sorted(value, key=lambda r: -r[-1])[:keep]
            out[f"{key}_count"] = len(value)
        else:
            out[key] = value
    return out


def post_fit_point(result: dict, row: dict | None) -> dict:
    """``post_fit_checks.point_inputs`` of one attempt from its fit row (cameras, ball
    observations and player boxes the fit itself read)."""
    from cv.pipeline import event_beam

    cameras = observations = None
    reach = None
    if row:
        if row.get("cameras"):
            cameras = _load(event_beam._resolve(row["cameras"]))
        if row.get("labels"):
            observations = _load(event_beam._resolve(row["labels"]))
        boxes = event_beam.player_boxes(row)
        if cameras and boxes:
            reach = lambda f, xyz: flight_validation.foot_reach(cameras, boxes, f, xyz)  # noqa: E731
    return post_fit_checks.point_inputs(
        result, cameras=cameras, observations=observations, reach=reach
    )


def apply_post_fit_checks(
    result: dict, flights: list[dict], row: dict | None, score: dict
) -> dict[str, int]:
    """``TENNIS_POST_FIT_CHECKS=on``: the checks on every still-validated flight (after
    ``validate_flights`` and the fit-only bounce rules). Every checked flight records the
    decision in ``post_fit_checks`` (which checks ran, passed, failed, abstained, and the
    measures); a failing one gets ``confirmed`` False and the reasons in ``validation.reasons``
    (best-effort tier, never repaired). Returns the tally."""

    def verdict_span(f: dict) -> tuple[float, float]:
        end = (f.get("trim") or {}).get("end_frame_before", f["end_frame"])
        return post_fit_checks.span_key(f["start_frame"], end)

    spans = [verdict_span(f) for f in flights if f.get("confirmed")]
    tally: dict[str, int] = {}
    if not spans:
        return tally
    point = {**post_fit_point(result, row), "score": score}
    decisions = post_fit_checks.evaluate(point, spans)
    for flight in flights:
        decision = decisions.get(verdict_span(flight)) if flight.get("confirmed") else None
        if decision is None:
            continue
        for name, check in decision["checks"].items():
            check["measures"] = _trim_measures(check["measures"])
            key = f"{name}:{check['status']}"
            tally[key] = tally.get(key, 0) + 1
        flight["post_fit_checks"] = decision
        tally["checked"] = tally.get("checked", 0) + 1
        if decision["demote"]:
            flight["confirmed"] = False
            validation = flight.setdefault("validation", {"ok": False, "reasons": [], "tags": []})
            validation["ok"] = False
            validation["reasons"] = list(validation.get("reasons") or []) + decision["reasons"]
            tally["demoted"] = tally.get("demoted", 0) + 1
    return tally


def _final_stage(attempt: dict, refits: dict, beam: dict, out: Path):
    """(stage folder, final source, fit row) of an attempt's final S6 result, or Nones."""
    refit = refits.get(attempt["key"])
    moved = (beam.get("changed") or {}).get(attempt["key"])
    if moved:
        stage = in_match(moved["result"], out).parent
        return stage, "event_beam", _load(in_match(moved["manifest"], out))["rows"][0]
    if refit and refit["status"] == "completed":
        folder = out / "refit" / attempt["key"]
        return folder / "stage", "cascade_refit", _load(folder / "job.json")["row"]
    if attempt["status"] == "completed":
        return attempt["stage"], "first_fit", _load(attempt["job"])["row"]
    return None, None, None


def post_fit_context(attempts: list[dict], refits: dict, beam: dict, out: Path) -> dict:
    """Score context per attempt key (``post_fit_checks.score_context``) with the server's end
    from the fitted serves of the same game (a pre-pass over the final results)."""
    context = post_fit_checks.score_context(
        [(a["key"], a.get("ledger_row") or {}) for a in attempts]
    )
    sides = []
    for attempt in attempts:
        stage, _, _ = _final_stage(attempt, refits, beam, out)
        if stage is None or not (stage / "result.json").is_file():
            continue
        side = post_fit_checks.serve_side(_load(stage / "result.json"))
        sides.append((context.get(attempt["key"], {}).get("game"), side))
    ends = post_fit_checks.server_sides(sides)
    for score in context.values():
        game = score.get("game")
        score["server_side"] = ends.get(tuple(game)) if game else None
    return context


def step_assemble(args) -> dict:
    summary, attempts = fit_attempts(args.out / "fit")
    refits = {}
    if (args.out / "refit" / "REFIT.json").is_file():
        refits = {r["attempt"]: r for r in _load(args.out / "refit" / "REFIT.json")["runs"]}
    beam = {}
    if (args.out / "beam" / "BEAM.json").is_file():
        beam = _load(args.out / "beam" / "BEAM.json")
    validation = getattr(args, "flight_validation", "on")
    # TENNIS_FLIGHT_PLAUSIBILITY=on adds two validation reasons; recorded only when on.
    plausibility = validation == "on" and flight_validation.plausibility_enabled()
    # TENNIS_FLIGHT_TRIM=on adds the trim/reject checks, and a flight failing validation leaves
    # the validated tier for the best-effort tiers with its reasons; recorded only when on.
    trim = validation == "on" and flight_validation.trim_enabled()
    failing: dict[str, int] = {}
    decisions = _load(args.out / "cascade" / "adjudicate" / "decisions.json")
    cost = _load(args.out / "cascade" / "COST.json")
    rows = []
    gates = product_outcomes.upstream_gates(args.out / "upstream")
    files = product_outcomes.StageFiles(args.out)
    outcomes, policy = [], None
    # Best-effort tier (TENNIS_BEST_EFFORT_TIER=on): separate rows and counts; when off, nothing
    # below differs from the validated-only document.
    best_effort = best_effort_tier.enabled()
    best_effort_slots, s6_complete = [], []
    # Fit-only bounce rules (TENNIS_FIT_ONLY_BOUNCE_RULES=on): demoted flights leave the
    # validated tier like trim-rejected ones; recorded only when on.
    rules_on = fit_only_bounce_rules.enabled()
    rules_tally: dict[str, int] = {}
    detector = None
    # Post-fit checks (TENNIS_POST_FIT_CHECKS=on): after the rules, failing flights leave the
    # validated tier the same way; recorded only when on.
    checks_on = post_fit_checks.enabled()
    checks_tally: dict[str, int] = {}
    context = post_fit_context(attempts, refits, beam, args.out) if checks_on else {}
    for attempt in attempts:
        refit = refits.get(attempt["key"])
        moved = (beam.get("changed") or {}).get(attempt["key"])
        stage, source, row = _final_stage(attempt, refits, beam, args.out)
        flights, point_end, result = [], None, None
        if stage:
            result = _load(stage / "result.json")
            flights = attempt_flights(result)
            # Post-processing only: flights after an automatic point-end call stay
            # emitted, marked ``after_point_end`` for consumers to drop or keep.
            point_end = point_end_tag.tag_result(result, flights)
            for flight in flights:
                flight["confirmed"] = True
            if validation == "on":
                for reason, n in validate_flights(result, row, flights).items():
                    failing[reason] = failing.get(reason, 0) + n
            if rules_on:
                if detector is None:
                    upstream = args.out / "upstream"
                    detector = fit_only_bounce_rules.DetectorRecord.from_bytes(
                        _load(upstream / "event_emissions.json"),
                        (upstream / "event_emissions.predictions.npz").read_bytes(),
                    )
                packet = attempt["stage"].parent / "inputs" / "packet.json"
                for key, n in apply_fit_only_bounce_rules(
                    result,
                    flights,
                    detector,
                    attempt["clip"],
                    packet if source == "first_fit" else None,
                ).items():
                    rules_tally[key] = rules_tally.get(key, 0) + n
            if checks_on:
                for key, n in apply_post_fit_checks(
                    result, flights, row, context.get(attempt["key"]) or {}
                ).items():
                    checks_tally[key] = checks_tally.get(key, 0) + n
        if policy is None and attempt["job"].is_file():
            policy = _load(attempt["job"]).get("policy")
        outcomes.append(
            product_outcomes.attempt_outcome(
                attempt,
                result,
                final_source=source,
                refit_status=refit["status"] if refit else None,
                final_flights=flights,
                point_end=point_end,
                gates=gates.get(attempt["clip"]),
                fit_job=product_outcomes.fit_job_record(files, attempt, args.out),
            )
        )
        if any(_demoted(f) for f in flights):
            # A point whose flight the rules or checks demoted is not complete in the
            # validated tier.
            outcomes[-1]["complete_point"] = False
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
                "event_beam_contacts": moved["contacts"] if moved else [],
                "final_source": source,
                "result": provenance.file_record(stage / "result.json") if stage else None,
                "accepted_flights": flights,
                "point_end_call": point_end,
            }
        )
        if best_effort:
            demoting = trim or rules_on or checks_on
            confirmed = [f for f in flights if f["confirmed"]] if demoting else flights
            slots = best_effort_tier.attempt_rows(result, confirmed, row) if result else []
            if demoting:
                rejected = {
                    round(float(f["start_frame"]), 1): f["validation"]["reasons"]
                    for f in flights
                    if not f["confirmed"] and f.get("validation")
                }
                for slot in slots:
                    reasons = rejected.get(round(float(slot["start_frame"]), 1))
                    if reasons:
                        slot["validation_rejected"] = reasons
            rows[-1]["best_effort_flights"] = slots
            best_effort_slots.append(slots)
            demoted = any(_demoted(f) for f in flights)
            s6_complete.append(
                bool(((result or {}).get("verdict") or {}).get("complete_point")) and not demoted
            )
    steps = {step: _load(args.out / "steps" / f"{step}.json") for step in FIT_STEPS}
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
            "confirmed_flights": sum(f["confirmed"] for r in rows for f in r["accepted_flights"]),
            "event_beam_changed_attempts": len(beam.get("changed") or {}),
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
        "event_beam": {
            "switch": getattr(args, "event_beam", "on"),
            "schema": beam.get("schema"),
            "candidates": beam.get("candidates"),
            "fits": beam.get("fits"),
            "attempts_changed": beam.get("attempts_changed"),
        },
        # Per-match audio/video delay (label-free); None when the beam is off.
        "audio_delay_calibration": beam.get("calibration"),
        "flight_validation": {
            "switch": validation,
            "schema": "flight_validation_v1" if validation == "on" else None,
            **({"plausibility": "on"} if plausibility else {}),
            **({"trim": "on"} if trim else {}),
            "failing_by_reason": failing,
            "note": "Flights failing validation stay listed with reasons; not confirmed.",
        },
        **(
            {
                "fit_only_bounce_rules": {
                    "switch": "on",
                    "rules_version": fit_only_bounce_rules.RULES_VERSION,
                    "tally": rules_tally,
                    "note": "Flights demoted by the fit-only bounce rules stay listed with "
                    "confirmed False and their evidence; fitted states past a flight's verdict "
                    "end (modelled continuation) are cut from the records.",
                }
            }
            if rules_on
            else {}
        ),
        **(
            {
                "post_fit_checks": {
                    "switch": "on",
                    "checks_version": post_fit_checks.CHECKS_VERSION,
                    "active": list(post_fit_checks.ACTIVE),
                    "tally": checks_tally,
                    "note": "Flights failing an active post-fit check stay listed with "
                    "confirmed False, their reasons and the evidence in post_fit_checks "
                    "(best-effort tier, not repaired).",
                }
            }
            if checks_on
            else {}
        ),
        "yield": None,
        "yield_note": "No labels exist for this broadcast; no reference-flight yield is claimed.",
        "attempts": rows,
    }
    extra = None
    if best_effort:
        layer = best_effort_tier.summary(best_effort_slots, len(attempts), s6_complete)
        document = {
            **{k: v for k, v in document.items() if k != "attempts"},
            "best_effort": layer,
            "attempts": rows,
        }
        extra = {"best_effort": layer}
    _save(args.out / "MATCH_RESULT.json", document)
    extra = {**(extra or {}), **failure_records(args, outcomes, document)} or None
    product_outcomes.write(
        args.out,
        args.match_id,
        attempts=outcomes,
        video=str(args.video),
        policy=policy,
        code=document["code"],
        extra=extra,
    )
    return document["counts"]


def failure_records(args, outcomes: list[dict] | None, match_result: dict | None) -> dict:
    """``FAILED_FLIGHTS.json`` and ``MISSED_PLAY.json`` (``TENNIS_FAILURE_INDEX``, default on).

    Review artifacts only: flights and every other output are unchanged, and an error here is
    recorded in ``OUTCOMES.json`` instead of failing the match.
    """
    if not failure_index.enabled():
        return {}
    records = {}
    try:
        document = failure_index.write(
            args.out, args.match_id, attempts=outcomes or [], match_result=match_result
        )
        records["failed_flights"] = failure_index.summary(document)
    except Exception as error:  # noqa: BLE001 - review artifact; never fails a match
        records["failed_flights"] = {"error": f"{type(error).__name__}: {error}"}
    try:
        records["missed_play"] = missed_play.summary(missed_play.write(args.out, args.match_id))
    except Exception as error:  # noqa: BLE001 - review artifact; never fails a match
        records["missed_play"] = {"error": f"{type(error).__name__}: {error}"}
    return records


def step_portal(args) -> dict:
    """Stage the final flights for the 3D portal (never the live portal directly).

    First fits go through the automatic exporter; each cascade refit then replaces its
    attempt's entry, exported from the edited packet and its own producer provenance.
    """
    from cv.pipeline import player_names
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
    finals = [
        (
            job["attempt"],
            args.out / "refit" / job["attempt"] / "stage" / "result.json",
            in_match(job["manifest"], args.out),
            "cascade_refit",
            f"after {adjudicated}",
        )
        for job in wave["jobs"]
    ]
    beam_path = args.out / "beam" / "BEAM.json"
    changed = (_load(beam_path).get("changed") or {}) if beam_path.is_file() else {}
    # A beam fit supersedes its attempt's refit (it was rebound on the same decisions).
    finals = [f for f in finals if f[0] not in changed] + [
        (
            key,
            in_match(c["result"], args.out),
            in_match(c["manifest"], args.out),
            "event_beam",
            "after the event beam",
        )
        for key, c in changed.items()
    ]
    beamed = []
    for attempt, result_path, manifest_path, source, suffix in finals:
        if not result_path.is_file():
            continue
        row = _load(manifest_path)["rows"][0]
        try:
            entry = shared.export_case(
                row, result_path, manifest_path, staging, root, repo, f"{label} · {suffix}"
            )
        except Exception as error:  # noqa: BLE001 - recorded; the earlier fit stays shown
            failed.append({"attempt": attempt, "error": f"{type(error).__name__}: {error}"})
            continue
        entry["product_final_source"] = source
        index["points"] = [p for p in index["points"] if p.get("key") != entry.get("key")] + [entry]
        (beamed if source == "event_beam" else replaced).append(attempt)
    index["count"] = len(index["points"])
    legacy._write_json(index_path, index, pretty=True)
    if best_effort_tier.enabled():
        layer = best_effort_tier.portal_layer(_load(args.out / "MATCH_RESULT.json"))
        legacy._write_json(staging / "data" / "best_effort.json", layer, pretty=False)
    names = {"skipped": f"{player_names.ENV}=off"}
    if player_names.enabled():
        try:
            # Identity per attempt into PLAYER_IDENTITY.json, MATCH_RESULT and the documents.
            names = player_names.run(args.out, args.match_id, root / player_names.METADATA, staging)
        except Exception as error:  # noqa: BLE001 - recorded; the viewer falls back to near/far
            names = {"error": f"{type(error).__name__}: {error}"}
    return {
        "staging": str(staging),
        "player_names": names,
        "exported_attempts": report["exported_attempts"],
        "held_before_fitting": len(report["held_before_fitting"]),
        "native_clock_holds": report.get("native_clock_holds", []),
        "cascade_refits_replaced": replaced,
        "event_beam_replaced": beamed,
        "cascade_refit_export_failures": failed,
    }


def step_pose(args) -> dict:
    """3D player pose for every staged attempt window (``player_pose_3d``), before ``compact``.

    Reads the native frame folders ``compact`` deletes, writes ``OUT/pose`` and the portal's
    per-clip pose layers bound into each document (``player_pose_portal``). ``TENNIS_PLAYER_POSE_3D=off`` skips it
    (rollback). It runs under :mod:`cv.pipeline.pose_gpu`: one pose step at a time per box, after
    the card has the pose's free memory, a CUDA out-of-memory attempt retried. When every attempt
    ran out of memory the step fails (the match is not published without pose); any other
    failure is recorded and does not fail the match: flights stay published.
    """
    from cv.pipeline import player_pose_3d

    if not player_pose_3d.enabled():
        return {"skipped": f"{player_pose_3d.ENV}=off"}
    staging = args.out / "portal" / "3d"
    if not (staging / "data").is_dir():
        return {"skipped": "no staged portal"}
    if not (args.out / "upstream" / args.match_id / "audit_frames_native_1080").is_dir():
        # Already compacted (a rerun of a finished match): no native pictures to pose.
        return {"skipped": "native frames already compacted"}
    command = [
        sys.executable,
        "-m",
        "cv.pipeline.player_pose_3d",
        "--match-out",
        str(args.out / "upstream" / args.match_id),
        "--portal",
        str(staging),
        "--out",
        str(args.out / "pose"),
        "--match-id",
        args.match_id,
        "--device",
        "0",
    ]
    log = args.out / "logs" / "pose.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)}
    saved = {key: os.environ.get(key) for key in (gpu_lease.ENV_SPEC, gpu_lease.ENV_LOG)}
    if getattr(args, "gpu_lease", None):
        # gpu_lease.run reads the specification from this process's environment.
        os.environ[gpu_lease.ENV_SPEC] = args.gpu_lease
        os.environ[gpu_lease.ENV_LOG] = str(args.out / "logs" / "gpu_leases.jsonl")
        env.pop(gpu_lease.ENV_HELD, None)
    leased = bool(getattr(args, "gpu_lease", None))
    guard = pose_gpu.settings()

    def attempt() -> tuple[int, str]:
        with log.open("a") as handle:
            handle.write(f"$ {' '.join(command)}\n")
            handle.flush()
            offset = handle.tell()
            completed = gpu_lease.run(
                command,
                stage="player_pose",
                # A leased card is chosen by the lease, with the pose need as its memory check.
                need_gb=guard["free_gb"] if leased else None,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        with log.open("rb") as handle:
            handle.seek(offset)
            return completed.returncode, handle.read().decode(errors="replace")

    try:
        # At most one pose step per box, free memory first, OOM retried (cv.pipeline.pose_gpu).
        returncode, attempts, guard = pose_gpu.guarded_run(
            attempt,
            gpu=None if leased else int(args.gpu),
            config=guard,
            record=args.out / "logs" / "pose_attempts.jsonl",
        )
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    guarded = {"gpu_guard": guard, "attempts": attempts}
    receipt = args.out / "pose" / player_pose_3d.RECEIPT_NAME
    if returncode != 0 or not receipt.is_file():
        return {
            "status": "failed",
            "returncode": returncode,
            "error": _last_error(log),
            **guarded,
        }
    document = _load(receipt)
    return {
        "status": "done",
        **guarded,
        "receipt": provenance.file_record(receipt),
        "counts": document.get("counts"),
        "documents": document.get("documents"),
        "timing": document.get("timing"),
        "npz_bytes": (args.out / "pose" / player_pose_3d.NPZ_NAME).stat().st_size,
    }


def step_compact(args) -> dict:
    """Overlay layer, then bounded per-match disk: portal clips, archives, bulk deleted."""
    from cv.pipeline import product_artifacts

    overlay = step_flight_overlay(args)
    fit_only_documents = step_fit_only_documents(args)
    document = product_artifacts.compact(
        args.out,
        args.match_id,
        frames=args.portal_frames,
        frame_attempts=args.portal_frame_attempts,
        keep=args.keep,
        workers=args.workers,
    )
    if failure_index.enabled():
        # The bundle and the archives exist now: the index's readiness follows them.
        try:
            index = failure_index.refresh(args.out, args.match_id)
            if index is not None:
                product_outcomes.write(
                    args.out, args.match_id, extra={"failed_flights": failure_index.summary(index)}
                )
        except Exception as error:  # noqa: BLE001 - review artifact; never fails a match
            product_outcomes.write(
                args.out,
                args.match_id,
                extra={"failed_flights": {"error": f"{type(error).__name__}: {error}"}},
            )
    return {
        "keep": args.keep,
        "total_before": document["total_before"],
        "total_after": document["total_after"],
        "flight_overlay": overlay,
        **({"fit_only_bounce_documents": fit_only_documents} if fit_only_documents else {}),
    }


def step_fit_only_documents(args) -> dict | None:
    """``TENNIS_FIT_ONLY_BOUNCE_RULES=on``: the point documents follow the assembled decisions
    (``fit_only_bounce_documents.label_document``: continuation past each validated end
    labelled, demoted flights best-effort). Runs inside ``compact`` after the overlay layer, so
    the overlay's fitted paths are cut too. A failure is recorded and does not fail the match.
    """
    from cv.pipeline import fit_only_bounce_documents
    from cv.viz import export_connected_3d as legacy

    if not (fit_only_bounce_rules.enabled() or post_fit_checks.enabled()):
        return None
    data = args.out / "portal" / "3d" / "data"
    if not data.is_dir() or not (args.out / "MATCH_RESULT.json").is_file():
        return {"skipped": "no staged portal or match result"}
    try:
        tally = {"documents": 0, "changed": 0, "demoted_flights": 0, "continuation_frames": 0}
        for attempt in _load(args.out / "MATCH_RESULT.json").get("attempts") or []:
            demoted = {
                round(float(f["start_frame"]), 4): _demotion_reasons(f)
                for f in attempt.get("accepted_flights") or []
                if _demoted(f)
            }
            key = attempt["key"]
            paths = [data / f"automatic_s6_{key}.json"]
            paths += sorted(data.glob(f"automatic_s6_{key}__component_*.json"))
            loaded = {path: _load(path) for path in paths if path.is_file()}
            # The overlay lives in the attempt document, so every document of a pieced point is
            # labelled with the states of all its pieces (as the post-hoc retier does).
            states: dict = {}
            for document in loaded.values():
                states.update(fit_only_bounce_documents.flight_states(document))
            for path, document in loaded.items():
                document, summary = fit_only_bounce_documents.label_document(
                    document, demoted, states
                )
                tally["documents"] += 1
                if summary.get("changed"):
                    legacy._write_json(path, document)
                    tally["changed"] += 1
                    tally["demoted_flights"] += len(summary["demoted_flights"])
                    tally["continuation_frames"] += summary["continuation_frames"]
        return {"status": "done", "rules_version": fit_only_bounce_rules.RULES_VERSION, **tally}
    except Exception as error:  # noqa: BLE001 - display layer; never fails a match
        return {"status": "failed", "error": f"{type(error).__name__}: {error}"}


def step_flight_overlay(args) -> dict:
    """Court, boxes, 2D skeletons and fits on the overlay video (``portal_flight_overlay``).

    Runs inside ``compact`` before the fitter cameras are archived. A failure is recorded and
    does not fail the match: the documents keep the ball-only overlay.
    """
    from cv.pipeline import portal_flight_overlay

    if not portal_flight_overlay.enabled():
        return {"skipped": f"{portal_flight_overlay.ENV}=off"}
    staging = args.out / "portal" / "3d"
    if not (staging / "data").is_dir():
        return {"skipped": "no staged portal"}
    if not (args.out / "fit" / "attempts").is_dir():
        return {"skipped": "fitter inputs already compacted"}
    try:
        summary = portal_flight_overlay.fill(
            staging, portal_flight_overlay.MatchInputs(args.out, args.match_id)
        )
    except Exception as error:  # noqa: BLE001 - display layer; never fails a match
        return {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    summary.pop("rows")
    return {"status": "done", **summary}


STEPS = ("upstream", "fit", "cascade", "refit", "beam", "assemble", "portal", "pose", "compact")
#: The steps whose time ``MATCH_RESULT.json`` reports (it is written by ``assemble``).
FIT_STEPS = STEPS[:5]


#: Steps that still run for a match whose upstream declared no processable attempts.
DECLARED_EMPTY_RUNS = ("upstream", "compact")


def declared_no_attempts(args) -> Path | None:
    """Upstream's declared record when the automatic point ledger is empty."""
    from cv.pipeline.broadcast_runner import NO_PROCESSABLE_ATTEMPTS_NAME

    path = args.out / "upstream" / args.match_id / NO_PROCESSABLE_ATTEMPTS_NAME
    return path if path.is_file() else None


def step_declared_empty(args, step: str, declared: Path) -> dict:
    """A step with nothing to do: no attempts were found, so nothing is fitted or exported.

    ``assemble`` still writes ``MATCH_RESULT.json`` and ``OUTCOMES.json`` (0 attempts, 0
    flights, the declared reason), so the match ends ``done`` with a reason instead of failed.
    """
    record = _load(declared)
    detail = {
        "skipped": "no_processable_attempts",
        "reason": record.get("reason"),
        "declared": provenance.file_record(declared),
    }
    if step != "assemble":
        return detail
    counts = {
        "ledger_attempts": 0,
        "attempts_including_split_children": 0,
        "first_fit_completed": 0,
        "held_or_failed_before_fit": 0,
        "attempts_with_flights": 0,
        "accepted_flights": 0,
        "accepted_flights_after_point_end": 0,
        "confirmed_flights": 0,
    }
    steps = {step: _load(args.out / "steps" / f"{step}.json") for step in FIT_STEPS}
    document = {
        "schema": SCHEMA,
        "match_id": args.match_id,
        "source_video": provenance.file_record(args.video),
        "code": provenance.git_record(REPO),
        "observation_origin": "automatic",
        "evaluation_labels_read": False,
        "no_processable_attempts": record,
        "counts": counts,
        "cost": {"paid_usd": 0.0, "by_model": {}, "paid_calls": 0},
        "time": {
            "wall_seconds": round(sum(s.get("wall_seconds") or 0 for s in steps.values()), 1),
            "cpu_seconds": round(sum(s.get("cpu_seconds") or 0 for s in steps.values()), 1),
            "gpu_seconds": round(sum(s.get("gpu_seconds") or 0 for s in steps.values()), 1),
        },
        "yield": None,
        "yield_note": "No processable attempts were found; nothing was fitted.",
        "attempts": [],
    }
    _save(args.out / "MATCH_RESULT.json", document)
    product_outcomes.write(
        args.out,
        args.match_id,
        attempts=[],
        video=str(args.video),
        code=document["code"],
        extra={
            "no_processable_attempts": {"reason": record.get("reason"), **detail},
            # A match without attempts is exactly what the missed-play flags are for.
            **failure_records(args, [], document),
        },
    )
    return {**detail, **counts}


def run(args) -> dict:
    """Every step in order; ``OUTCOMES.json`` is refreshed after each and on failure."""
    args.out.mkdir(parents=True, exist_ok=True)
    records = {}
    for step in STEPS:
        declared = declared_no_attempts(args) if step not in DECLARED_EMPTY_RUNS else None

        def body(step=step, declared=declared):
            if declared is not None:
                return step_declared_empty(args, step, declared)
            return globals()[f"step_{step}"](args)

        try:
            records[step] = metered(step, args.out, body)
        finally:
            product_outcomes.refresh(args.out, args.match_id)
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
        "--score-vlm-url",
        help="an already-running scoreboard VLM (same model) instead of starting one",
    )
    p.add_argument(
        "--qwen-ports",
        type=lambda text: tuple(int(x) for x in text.split(",") if x.strip()),
        default=(),
        help="already-running local cascade Qwen servers instead of starting them",
    )
    p.add_argument(
        "--portal-frames",
        choices=("video", "native", "none"),
        default="video",
        help="compact: portal pictures as one H.264 clip per attempt, as-is, or none",
    )
    p.add_argument(
        "--portal-frame-attempts",
        choices=("accepted", "all"),
        default="accepted",
        help="compact: keep pictures for attempts with accepted flights, or every fitted one",
    )
    p.add_argument(
        "--keep",
        choices=("standard", "lean", "full"),
        default="standard",
        help="compact: archive S6 stages (standard), delete them (lean), delete nothing (full)",
    )
    p.add_argument(
        "--gpu-lease",
        help="upstream: JSON lease spec (cv.pipeline.gpu_lease); GPU stages lease a GPU of "
        "this machine while they run instead of holding --gpu for the whole step "
        "(throughput only)",
    )
    p.add_argument(
        "--concurrent-tracking",
        action="store_true",
        help="upstream: run each tracking stage's passes concurrently from one decode per "
        "frame, co-resident on --gpu (throughput only; outputs identical)",
    )
    p.add_argument(
        "--long-shot-view-windows",
        action="store_true",
        help="upstream: in a continuous feed (a shot over 20 min), classify shots longer than "
        "3 min in ~20 s windows (default off)",
    )
    p.add_argument(
        "--scoreboard-map-agreement",
        action="store_true",
        help="upstream: locate the scoreboard on a component both overlay stability maps "
        "find (fixed-camera feeds; default off)",
    )
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
        choices=("qwen", "gemini", "none"),
        default=None,
        help=(
            "paid cascade first tier: local Qwen (FREEZE v2) or Gemini Flash (no Qwen servers); "
            "none: no paid model named, refused unless adjudication is fitter. "
            "Default: event_cascade.DEFAULT_SETTINGS"
        ),
    )
    p.add_argument(
        "--event-beam",
        choices=("on", "off"),
        default="on",
        help="time-witness contact beam after the refit (off: rollback, no beam fits)",
    )
    p.add_argument(
        "--event-beam-prune",
        choices=("off", "time_witness"),  # event_beam.PRUNE_MODES
        default="off",
        help="drop ranked beam branches without an audio/S5-peak time witness (default off)",
    )
    p.add_argument(
        "--flight-validation",
        choices=("on", "off"),
        default="on",
        help="post-validation of accepted flights at assemble (off: every accepted flight confirmed)",
    )
    p.add_argument("--until", choices=STEPS, help="stop after this step")
    return p


def main() -> None:
    from cv.pipeline import math_threads

    math_threads.apply()
    args = parser().parse_args()
    args.video = args.video.resolve()
    args.out = args.out.resolve()
    print(json.dumps(run(args), indent=2, default=str))


if __name__ == "__main__":
    main()
