"""Run shared local S6 from cached automatic broadcast artifacts.

This opt-in backend uses the original automatic point ledger, native extraction,
tracker, event, camera and player artifacts. It does not run upstream models or
read the legacy reconstruction. Each attempt executes in a separate process.

`load_policy` applies `s6_labeled_stage.PIPELINE_COMPONENT_POLICY`, including
production-default `whole_point_seed_fallback=on`. Rollback: delete that line
from `PIPELINE_COMPONENT_POLICY`. This entrypoint does not re-run player
association; it consumes pre-built sided boxes and refuses a sided CSV whose
`keep_unique_admissible_frames` sidecar flag does not match
`SelectionConfig` (production default on). Regenerate
`player_boxes_*_native_sided_v1.csv` via
`player_side_association --keep-unique-admissible-frames`. Rollback of the
tracker default: `SelectionConfig.keep_unique_admissible_frames = False`.

Usage: python -m cv.pipeline.s6_broadcast_backend --help
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from cv.pipeline import artifact_cache, paths, provenance, resolution
from cv.pipeline.source_timebase import audit_timestamps, frame_pts

REPO = Path(__file__).resolve().parents[2]
# Explicitly omit all legacy fitted-output and point-slicing receipts.
UPSTREAM_STAGES = (
    "broadcast_source",
    "play_camera",
    "score_reading",
    "serve_audio",
    "shot_boundaries",
    "native_view_frames",
    "frame_embeddings",
    "shot_views",
    "player_boxes",
    "score_state",
    "serve_attempts",
    "point_ledger",
    "postseg_materialization",
    "postseg_base",
    "player_side_association",
    "tracking",
    "point_gates",
    "event_inference",
    "physical_player_motion",
    "contact_striker",
)


#: Component-family policy forwarded into ordinary video inference.
#: `uncertain_original_occurrence` is forwarded but is NOT a trigger: on its own it
#: has no automatic consumer, and making it one would start forwarding
#: `contact_components` for policies that never forwarded it before -- a silent
#: change to every existing automatic receipt that declares it.
COMPONENT_TRIGGERS = (
    "leading_event_components",
    "contact_component_routing",
    "automatic_abstained_occurrence",
    # A trigger in its own right: `build_observations` consumes it directly, and it is
    # the only key here that is NOT an admission, so it must be able to reach ordinary
    # video inference even if nothing else is declared.  Adding it changes no existing
    # receipt, because no policy written before 2026-09-19 can carry it.
    "automatic_event_operating_point",
)
COMPONENT_FORWARDED = (*COMPONENT_TRIGGERS, "uncertain_original_occurrence")


def _read(path: Path):
    return json.loads(path.read_text())


def _save(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
    pending.replace(path)


def _identity(record: dict) -> tuple:
    return tuple(record.get(k) for k in ("path_base", "path", "sha256"))


def source_pts_cache(video: Path, fps: float, cache_root: Path) -> tuple[Path, Path]:
    """Decode actual source PTS once per source/configuration, with a bound receipt."""
    source = provenance.file_record(video, role="processing_video")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-nofind_stream_info",
        "-threads",
        "1",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=pts_time",
        "-of",
        "compact=p=1:nk=0",
        str(video),
    ]
    cadence = float(fps)
    identity = dict(
        source=source, fps=int(cadence) if cadence.is_integer() else cadence, command=command[:-1]
    )
    cache = cache_root / artifact_cache._digest_json(identity)
    cache.mkdir(parents=True, exist_ok=True)
    pts, receipt = cache / "source_pts.txt", cache / "receipt.json"
    with (cache / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if pts.is_file() and receipt.is_file():
            prior = _read(receipt)
            current_output = provenance.file_record(pts)
            previous_output = prior.get("output", {})
            relocated = (
                previous_output.get("path_base") == "unconfigured_external"
                and previous_output.get("path") == pts.name
                and all(previous_output.get(k) == current_output[k] for k in ("sha256", "bytes"))
            )
            if (
                prior.get("identity") == identity
                and (previous_output == current_output or relocated)
                and prior.get("audit", {}).get("nominal_timeline_valid") is True
            ):
                if relocated:
                    prior.update(previous_output_binding=previous_output, output=current_output)
                    _save(receipt, prior)
                return pts, receipt
            raise ValueError("source PTS cache identity or bytes changed")
        decode_started = time.monotonic()
        pending = pts.with_suffix(".pending")
        with pending.open("w") as output, (cache / "ffprobe.stderr").open("w+") as errors:
            completed = subprocess.run(command, stdout=output, stderr=errors, check=False)
            errors.seek(0)
            error_text = errors.read(4096).strip()
        if completed.returncode or error_text:
            pending.unlink(missing_ok=True)
            raise ValueError(f"source PTS decoder failed: {completed.returncode}: {error_text}")
        with pending.open() as handle:
            audit = audit_timestamps(frame_pts(handle), fps)
        if not audit["nominal_timeline_valid"]:
            pending.unlink(missing_ok=True)
            raise ValueError("source PTS inventory is incomplete or irregular")
        if provenance.file_record(video, role="processing_video") != source:
            pending.unlink(missing_ok=True)
            raise ValueError("source video changed during PTS extraction")
        pending.replace(pts)
        _save(
            receipt,
            dict(
                schema="s6_source_pts_cache_v1",
                identity=identity,
                output=provenance.file_record(pts),
                audit=audit,
                decoder_seconds=time.monotonic() - decode_started,
            ),
        )
    return pts, receipt


def load_policy(path: Path) -> dict:
    """Freeze a supplied composed policy; no per-case switches or ambient prior."""
    from cv.pipeline import s6_input_origin, s6_labeled_stage as stage

    policy = _read(path)
    if not isinstance(policy, dict):
        raise ValueError("shared S6 policy must be an explicit object")
    # The production automatic entrypoint declares the measured component admissions
    # explicitly; `shared_settings` keeps its strict defaults so every existing receipt
    # reproduces.  A supplied value always wins, and a policy without the unresolved-ending
    # prefix/component contracts is untouched.  See `stage.PIPELINE_COMPONENT_POLICY`.
    policy = stage.pipeline_policy(policy)
    required = {
        *stage.NUMERICAL_POLICY,
        "coarse_iterations",
        "refine_iterations",
        "search_seconds",
        "preparation",
        "refinement",
        "athlete_evidence",
        "serve_location_prior",
    }
    if missing := required - policy.keys():
        raise ValueError(f"shared S6 policy is incomplete: {sorted(missing)}")
    settings = stage.shared_settings(policy)
    allowed = required | settings.keys()
    if unknown := policy.keys() - allowed:
        raise ValueError(f"unsupported shared S6 policy fields: {sorted(unknown)}")
    if any(policy[k] != v for k, v in stage.NUMERICAL_POLICY.items()):
        raise ValueError("shared S6 numerical policy does not match the stage")
    if policy["athlete_evidence"] != "optional":
        raise ValueError("automatic backend requires optional athlete evidence")
    for key in ("coarse_iterations", "refine_iterations", "search_seconds"):
        if (
            type(policy[key]) not in (int, float)
            or not math.isfinite(policy[key])
            or policy[key] <= 0
        ):
            raise ValueError(f"positive finite {key} required")
    s6_input_origin.serve_prior_path({"observation_origin": "automatic"}, policy, stage.resolve)
    return {**policy, **settings}


def discover_upstream(upstream_root: Path, match_id: str) -> tuple[list[Path], list[Path]]:
    """Only pre-reconstruction ancestry; never follow the final legacy run manifest."""
    directory = upstream_root / match_id / "run_manifests"
    receipts = [directory / "stage_receipts" / f"{s}.json" for s in UPSTREAM_STAGES]
    receipts = [p for p in receipts if p.is_file()]
    parents = [
        upstream_root / "provenance.json",
        directory / "canonical_runner.json",
        directory / "broadcast_runner_point_ledger.json",
    ]
    return receipts, [p for p in parents if p.is_file()]


def _producer_bindings(receipts: list[Path], parents: list[Path], source_video: Path) -> set[tuple]:
    """Require genuine automatic producer identities before adding new S6 bindings."""
    if not receipts or not parents:
        raise ValueError("automatic upstream stage receipts and parent provenance are required")
    covered = set()
    source_identity = _identity(provenance.file_record(source_video))
    source_bound = False
    for path in parents:
        parent = _read(path)
        document = parent.get("provenance", parent)
        provenance.validate_provenance(document)
        if document["mode"] != provenance.AUTOMATIC_MODE:
            raise ValueError("upstream parent is not automatic")
        provenance.assert_automatic_document(document, context=str(path))
        source_bound |= any(_identity(r) == source_identity for r in document["source_videos"])
    if not source_bound:
        raise ValueError("automatic upstream parents do not bind the declared processing video")
    for path in receipts:
        receipt = _read(path)
        if receipt.get("schema") != artifact_cache.SCHEMA:
            raise ValueError("original upstream stage receipt schema required")
        identity = receipt["identity"]
        provenance.assert_automatic_document(identity, context=str(path))
        unsigned = {k: v for k, v in identity.items() if k != "fingerprint"}
        if artifact_cache._digest_json(unsigned) != identity.get("fingerprint"):
            raise ValueError("upstream stage receipt fingerprint changed")
        covered.update(_identity(r) for r in receipt["outputs"] if r.get("kind") == "file")
    return covered


def event_override_ancestry(
    events: Path, producer_path: Path, source_video: Path
) -> tuple[Path, Path]:
    """Accept event artifacts only through a bound automatic producer and matching stage."""
    from cv.pipeline.s6_automatic_observations import _checked_record

    producer = provenance.load_provenance(producer_path, require_automatic=True)
    provenance.assert_automatic_document(producer, context="automatic event override")
    source = _identity(provenance.file_record(source_video))
    if not any(_identity(r) == source for r in producer["source_videos"]):
        raise ValueError("event override producer belongs to another processing video")
    event_identity = _identity(provenance.file_record(events))
    if not any(_identity(r) == event_identity for r in producer["reused_artifacts"]):
        raise ValueError("event override producer does not bind the executed emissions")
    stages = {
        "automatic_qualification_receipt": "event_cached_qualification",
        "automatic_event_inference_receipt": "event_inference",
    }
    records = [r for r in producer["reused_artifacts"] if r.get("role") in stages]
    if len(records) != 1:
        raise ValueError("event override needs one bound automatic event stage receipt")
    receipt = _checked_record(records[0], source_video)
    document = _read(receipt)
    if (
        document.get("schema") != artifact_cache.SCHEMA
        or document.get("identity", {}).get("stage") != stages[records[0]["role"]]
    ):
        raise ValueError("event override receipt role does not match its automatic producer stage")
    covered = _producer_bindings([receipt], [producer_path], source_video)
    if event_identity not in covered:
        raise ValueError("event stage receipt does not bind the executed emissions")
    return receipt, producer_path


def player_override_ancestry(
    players: Path, producer_path: Path, source_video: Path
) -> tuple[Path, Path]:
    """Admit an explicit automatic player producer, including its coordinate domain."""
    from cv.pipeline.s6_automatic_observations import _checked_record

    producer = provenance.load_provenance(producer_path, require_automatic=True)
    provenance.assert_automatic_document(producer, context="automatic player override")
    source = _identity(provenance.file_record(source_video))
    if not any(_identity(r) == source for r in producer["source_videos"]):
        raise ValueError("player override producer belongs to another processing video")
    required = [players, resolution.coordinate_manifest_path(players)]
    identities = {_identity(provenance.file_record(p)) for p in required}
    if not identities.issubset({_identity(r) for r in producer["reused_artifacts"]}):
        raise ValueError("player override producer must bind observations and coordinate sidecar")
    records = [
        r
        for r in producer["reused_artifacts"]
        if r.get("role") == "automatic_player_inference_receipt"
    ]
    if len(records) != 1:
        raise ValueError("player override needs one bound automatic player inference receipt")
    receipt = _checked_record(records[0], source_video)
    document = _read(receipt)
    if (
        document.get("schema") != artifact_cache.SCHEMA
        or document.get("identity", {}).get("stage") != "player_inference"
    ):
        raise ValueError("player override receipt role does not match its automatic producer stage")
    covered = _producer_bindings([receipt], [producer_path], source_video)
    if not identities.issubset(covered):
        raise ValueError("player stage receipt must bind observations and coordinate sidecar")
    return receipt, producer_path


def camera_override_ancestry(
    cameras: Path, producer_path: Path, source_video: Path
) -> tuple[Path, Path]:
    """Admit an explicit automatic camera producer, binding its actual projected camera rows."""
    from cv.pipeline.s6_automatic_observations import _checked_record

    producer = provenance.load_provenance(producer_path, require_automatic=True)
    provenance.assert_automatic_document(producer, context="automatic camera override")
    source = _identity(provenance.file_record(source_video))
    if not any(_identity(r) == source for r in producer["source_videos"]):
        raise ValueError("camera override producer belongs to another processing video")
    required = [cameras]
    identities = {_identity(provenance.file_record(p)) for p in required}
    if not identities.issubset({_identity(r) for r in producer["reused_artifacts"]}):
        raise ValueError("camera override producer must bind projected camera artifact")
    records = [
        r
        for r in producer["reused_artifacts"]
        if r.get("role") == "automatic_camera_inference_receipt"
    ]
    if len(records) != 1:
        raise ValueError("camera override needs one bound automatic camera inference receipt")
    receipt = _checked_record(records[0], source_video)
    document = _read(receipt)
    if (
        document.get("schema") != artifact_cache.SCHEMA
        or document.get("identity", {}).get("stage") != "window_camera_inference"
    ):
        raise ValueError("camera override receipt role does not match its automatic producer stage")
    covered = _producer_bindings([receipt], [producer_path], source_video)
    if not identities.issubset(covered):
        raise ValueError("camera stage receipt must bind projected camera artifact")
    return receipt, producer_path


def optional_pose_ancestry(
    pose: Path, producer_path: Path, source_video: Path
) -> tuple[Path, Path]:
    """Bind independent pose evidence without replacing the original actor stream."""
    from cv.pipeline.s6_automatic_observations import _checked_record

    producer = provenance.load_provenance(producer_path, require_automatic=True)
    provenance.assert_automatic_document(producer, context="independent contact pose")
    if _identity(provenance.file_record(source_video)) not in {
        _identity(r) for r in producer["source_videos"]
    }:
        raise ValueError("independent pose producer belongs to another processing video")
    identities = {
        _identity(provenance.file_record(p))
        for p in (pose, resolution.coordinate_manifest_path(pose))
    }
    if not identities.issubset({_identity(r) for r in producer["reused_artifacts"]}):
        raise ValueError("independent pose producer must bind CSV and coordinate sidecar")
    records = [
        r
        for r in producer["reused_artifacts"]
        if r.get("role") == "automatic_pose_inference_receipt"
    ]
    if len(records) != 1:
        raise ValueError("independent pose needs one bound automatic pose inference receipt")
    receipt = _checked_record(records[0], source_video)
    document = _read(receipt)
    if (
        document.get("schema") != artifact_cache.SCHEMA
        or document.get("identity", {}).get("stage") != "pose_player_crop"
    ):
        raise ValueError("independent pose receipt does not identify the actual pose producer")
    if not identities.issubset(_producer_bindings([receipt], [producer_path], source_video)):
        raise ValueError("pose stage receipt must bind CSV and coordinate sidecar")
    return receipt, producer_path


def prepare_provenance(
    *, inputs, pts_receipt: Path, receipts: list[Path], parents: list[Path], output: Path
) -> Path:
    """Bind original inputs, while keeping historical producers and fresh PTS distinct."""
    primary = [
        inputs.manifest,
        inputs.point_ledger,
        inputs.ball,
        inputs.events,
        inputs.cameras,
        inputs.players,
    ]
    # A scoped consumer's selection is an automatic materialization output too: require
    # its actual receipt binding, rather than trusting an otherwise plausible marker.
    from cv.pipeline.evaluation_scope import scope_path

    declared_scope = scope_path(Path(inputs.manifest).parent)
    if declared_scope.is_file():
        primary.append(declared_scope)
    if getattr(inputs, "optional_contact_pose", None) is not None:
        primary.append(inputs.optional_contact_pose)
    if getattr(inputs, "optional_contact_views", None) is not None:
        primary.append(inputs.optional_contact_views)
    covered = _producer_bindings(receipts, parents, inputs.source_video)
    if missing := [p.name for p in primary if _identity(provenance.file_record(p)) not in covered]:
        raise ValueError(f"automatic primary inputs are not producer-bound: {missing}")
    supplemental = [
        inputs.extraction_receipt,
        inputs.source_pts,
        pts_receipt,
        resolution.coordinate_manifest_path(inputs.ball),
        resolution.coordinate_manifest_path(inputs.players),
    ]
    if getattr(inputs, "optional_contact_pose", None) is not None:
        from cv.pipeline.s6_optional_contacts import independent_pose_frame_receipt

        supplemental.append(resolution.coordinate_manifest_path(inputs.optional_contact_pose))
        if pose_receipt := independent_pose_frame_receipt(inputs):
            supplemental.append(pose_receipt)
    ball_sidecar = _read(resolution.coordinate_manifest_path(inputs.ball))
    source = Path(str(ball_sidecar.get("source", "")).split(" + ")[0])
    if not source.is_absolute():
        source = inputs.ball.parent / source
    if source != inputs.ball and source.is_file() and source.suffix == ".csv":
        supplemental.extend([source, resolution.coordinate_manifest_path(source)])
    # Retain explicit event producer identity when supplied by the normal runner.
    event_manifest = inputs.events.with_name(inputs.events.stem + ".manifest.json")
    if event_manifest.is_file():
        supplemental.append(event_manifest)
    for path in [*primary, *supplemental]:
        if path.suffix == ".json":
            provenance.assert_automatic_document(_read(path), context=str(path))
    document = provenance.build_provenance(
        root=REPO,
        mode=provenance.AUTOMATIC_MODE,
        source_videos=[provenance.file_record(inputs.source_video, role="processing_video")],
        configuration=dict(
            entrypoint="cv.pipeline.s6_broadcast_backend",
            stage="automatic_pre_reconstruction_admission",
            match_id=inputs.match_id,
            clip=inputs.clip,
            supplemental_bindings="original artifacts admitted at S6 preparation",
        ),
        reused_artifacts=[
            *[provenance.file_record(p, role="automatic_stage_receipt") for p in receipts],
            *[provenance.file_record(p, role="automatic_parent_provenance") for p in parents],
            *[provenance.file_record(p, role="automatic_upstream_input") for p in primary],
            *[provenance.file_record(p, role="automatic_supplemental_input") for p in supplemental],
        ],
    )
    return provenance.write_provenance(output, document)


#: Statuses the shared executor writes for an attempt that produced no fit.
EXECUTOR_FAILURES = ("preparation_failed", "component_source_failed", "execution_failed", "timeout")


def needs_orchestration(policy: dict) -> bool:
    """Component policy and the whole-point seed fallback need the shared executor."""
    return (
        policy.get("contact_components", "off") != "off"
        or policy.get("whole_point_seed_fallback", "off") == "on"
    )


def run_orchestrated_stage(job: Path, output: Path, timeout_seconds: float) -> dict:
    """Run the attempt through the same executor as the labeled harness.

    ``output`` becomes a relative link to the executor's case directory, so consumers
    keep reading ``<output>/result.json`` and ``<output>/component_plan.json``.
    """
    from cv.pipeline import s6_component_orchestration as orchestration
    from cv.pipeline.s6_attempt_execution import execute

    started = time.monotonic()
    try:
        if output.exists() or output.is_symlink():
            raise ValueError("shared S6 needs a fresh attempt output directory")
        document = _read(job)
        row, policy = document["row"], document["policy"]
        run = output.with_name(f"{output.name}_run")
        if run.exists():
            raise ValueError("shared S6 needs a fresh executor directory")
        for folder in ("cases", "jobs", "logs"):
            (run / folder).mkdir(parents=True)
        execute(row, run, policy, timeout_seconds)
        output.symlink_to(Path(run.name) / "cases" / row["key"])
        result = _read(output / "result.json")
        if result.get("status") in EXECUTOR_FAILURES:
            return dict(
                status=result["status"],
                reason=result.get("reason"),
                seconds=time.monotonic() - started,
            )
        if result.get("key") != row["key"]:
            raise ValueError("shared S6 output does not bind its automatic attempt")
        if result.get("schema") != orchestration.SCHEMA and (
            result.get("observation_origin") != "automatic"
            or result.get("human_derived_inputs") is not False
            or result.get("upstream_automatic_inference_eligible") is not True
            or result.get("automatic_provenance") != row["automatic_provenance"]
        ):
            raise ValueError("shared S6 output does not bind its automatic attempt")
        return dict(
            status="completed",
            verdict=result["verdict"],
            result=provenance.file_record(output / "result.json"),
            seconds=time.monotonic() - started,
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        return dict(
            status="execution_failed",
            reason=f"{type(error).__name__}: {error}",
            seconds=time.monotonic() - started,
        )


def run_stage(job: Path, output: Path, timeout_seconds: float) -> dict:
    """Mutable numerical state never crosses attempt process boundaries."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if needs_orchestration(_read(job).get("policy", {})):
        return run_orchestrated_stage(job, output, timeout_seconds)
    command = [
        sys.executable,
        "-m",
        "cv.pipeline.s6_labeled_stage",
        "--job",
        str(job),
        "--output",
        str(output),
    ]
    started = time.monotonic()
    try:
        if output.exists():
            raise ValueError("shared S6 needs a fresh attempt output directory")
        with (output.parent / f"{output.name}.log").open("w") as log:
            completed = subprocess.run(
                command,
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
            )
        if completed.returncode:
            raise RuntimeError(f"shared S6 subprocess exited {completed.returncode}")
        result = _read(output / "result.json")
        expected = _read(job)["row"]
        if (
            result.get("observation_origin") != "automatic"
            or result.get("human_derived_inputs") is not False
            or result.get("upstream_automatic_inference_eligible") is not True
            or result.get("key") != expected["key"]
            or result.get("automatic_provenance") != expected["automatic_provenance"]
        ):
            raise ValueError("shared S6 output does not bind its automatic attempt")
        return dict(
            status="completed",
            verdict=result["verdict"],
            result=provenance.file_record(output / "result.json"),
            seconds=time.monotonic() - started,
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        subprocess.TimeoutExpired,
    ) as error:
        return dict(
            status="execution_failed",
            reason=f"{type(error).__name__}: {error}",
            seconds=time.monotonic() - started,
        )


#: Statuses held by full-ledger rows this run deliberately did not attempt.  They stay in the
#: roster and in every full-ledger count, and they are excluded from every *selected* count --
#: including the child denominator, where an unselected parent must not contribute a phantom
#: selected child.
UNSELECTED_STATUSES = ("not_run_cap", "not_run_scope")


def summarize(rows: list[dict], selected: int) -> dict:
    verdicts = [r.get("verdict", {}) for r in rows]
    unselected = set(UNSELECTED_STATUSES)
    complete = sum(bool(v.get("complete_point")) for v in verdicts)
    accepted = sum(v.get("accepted_flight_count", 0) for v in verdicts)
    emitted = sum(v.get("flight_count", 0) for v in verdicts)
    return dict(
        automatic_ledger_attempts=len(rows),
        selected_attempts=selected,
        completed_stage_attempts=sum(r["status"] == "completed" for r in rows),
        gate_accepted_complete_attempts=complete,
        complete_attempt_gate_percent=100 * complete / selected if selected else None,
        complete_roster_gate_lower_bound_percent=100 * complete / len(rows) if rows else None,
        not_run_cap_attempts=sum(r["status"] == "not_run_cap" for r in rows),
        not_run_scope_attempts=sum(r["status"] == "not_run_scope" for r in rows),
        full_roster_evaluated=selected == len(rows) and all(r["status"] != "pending" for r in rows),
        gate_accepted_emitted_flights=accepted,
        emitted_flights=emitted,
        emitted_flight_gate_percent=100 * accepted / emitted if emitted else None,
        reference_flights=None,
        reference_flight_yield_percent=None,
        visually_checked_useful_complete_attempts=None,
        useful_complete_yield_percent=None,
        incorrect_accept_count=None,
        denominator_note=(
            "pilot denominator is selected attempts including holds/failures; "
            "full ledger roster also retains deliberately unrun capped and "
            "out-of-scope attempts"
        ),
        flight_note="emitted-flight acceptance is not reference-flight recovery yield",
        **(
            {
                "split_parent_count": sum(bool(r.get("children")) for r in rows),
                "all_selected_child_attempts": sum(
                    len(r.get("children", [r])) for r in rows if r["status"] not in unselected
                ),
                "censored_child_aftermaths": sum(
                    c.get("service_attempt_scope", {}).get("aftermath", {}).get("kind")
                    == "identity_censored"
                    for r in rows
                    for c in r.get("children", [])
                ),
                "child_gate_accepts": sum(
                    c.get("verdict", {}).get("complete_point", False)
                    for r in rows
                    for c in r.get("children", [])
                ),
                "parent_coverage_note": "Every original selected parent and all its children retained; child gates are not reference recovery or full aftermath certification.",
            }
            if any(r.get("children") for r in rows)
            else {}
        ),
    )


def run_cached(
    *,
    upstream_root: Path,
    match_id: str,
    source_video: Path,
    policy_path: Path,
    output: Path,
    maximum: int | None = None,
    workers: int = 1,
    timeout_seconds: float = 4200,
    camera_name: str = "camera_P_per_frame_v1.npz",
    execution_scope: str = "cached_automatic_upstream",
    event_emissions: Path | None = None,
    event_provenance: Path | None = None,
    player_observations: Path | None = None,
    player_provenance: Path | None = None,
    service_attempt_split: str = "off",
    observed_first_flight: str = "off",
    camera_projections: Path | None = None,
    camera_provenance: Path | None = None,
    optional_contact_pose: Path | None = None,
    optional_contact_pose_provenance: Path | None = None,
    evaluation_scope: Path | None = None,
) -> Path:
    from dataclasses import replace
    from cv.pipeline import evaluation_scope as scope_module
    from cv.pipeline.s6_automatic_observations import AutomaticInputs, build_observations

    if service_attempt_split not in {"off", "on"}:
        raise ValueError("explicit service-attempt split policy required")
    if observed_first_flight not in {"off", "on"}:
        raise ValueError("explicit observed-first-flight policy required")
    if Path(match_id).name != match_id or not match_id:
        raise ValueError("safe match identity required")
    if workers < 1 or (maximum is not None and maximum < 1) or timeout_seconds <= 0:
        raise ValueError("positive execution limits required")
    if (event_emissions is None) != (event_provenance is None):
        raise ValueError("event emissions override requires its automatic producer provenance")
    if (player_observations is None) != (player_provenance is None):
        raise ValueError("player observations override requires its automatic producer provenance")
    if (camera_projections is None) != (camera_provenance is None):
        raise ValueError("camera projections override requires its automatic producer provenance")
    if (optional_contact_pose is None) != (optional_contact_pose_provenance is None):
        raise ValueError("independent contact pose requires its automatic producer provenance")
    policy = load_policy(policy_path)
    if optional_contact_pose is not None and policy.get("optional_contacts", "off") != "on":
        raise ValueError("independent contact pose requires optional contacts enabled")
    if (
        policy.get("optional_final_contacts", "off") != "off"
        and policy.get("optional_contacts", "off") != "on"
    ):
        raise ValueError("optional final contacts require the optional contact policy")
    if (
        policy.get("optional_contact_timing", "off") != "off"
        and policy.get("optional_contacts", "off") != "on"
    ):
        raise ValueError("optional contact timing requires optional contacts enabled")
    if (
        policy.get("optional_contact_composition", "off") != "off"
        and policy.get("optional_contacts", "off") != "on"
    ):
        raise ValueError("optional contact composition requires optional contacts enabled")
    if (
        policy.get("optional_interior_contacts", "off") != "off"
        and policy.get("optional_contacts", "off") != "on"
    ):
        raise ValueError("optional interior contacts require the optional contact policy")
    manifest_path = upstream_root / "broadcast_manifest.json"
    matches = [r for r in _read(manifest_path)["matches"] if r["id"] == match_id]
    if len(matches) != 1:
        raise ValueError("one matching automatic manifest entry required")
    match, directory = matches[0], upstream_root / match_id
    from cv.pipeline.player_side_association import require_keep_unique_admissible_frames

    players_path = (
        player_observations
        or directory / f"player_boxes_{float(match['source_fps']):g}_native_sided_v1.csv"
    )
    require_keep_unique_admissible_frames(players_path)
    ledger = directory / "automatic_point_ledger_v1.csv"
    with ledger.open(newline="") as handle:
        ledger_rows = list(csv.DictReader(handle))
    ids = [int(r["pt"]) for r in ledger_rows]
    if len(set(ids)) != len(ids) or set(ids) != set(match["point_ids"]):
        raise ValueError("automatic ledger and manifest attempt identities disagree")
    # The full-ledger/full-manifest identity above stays strict: a declared execution scope
    # limits which of those rows this stage tries, never which rows the roster holds.  The
    # marker is read from the root itself, so a standalone invocation against a scoped
    # upstream root cannot silently try rows that root never materialized.
    scope, limit = scope_module.guard_consumer_root(
        upstream_root=upstream_root,
        match_id=match_id,
        ledger=ledger,
        rows=ledger_rows,
        maximum=maximum,
        source_video=source_video,
    )
    if evaluation_scope is not None:
        if scope is None or Path(evaluation_scope) != scope_module.scope_path(upstream_root):
            raise ValueError("forwarded execution scope is not this upstream root's scope")
    if output.exists():
        raise FileExistsError("use a new S6 output directory; upstream cache remains reusable")
    paths.data_relative(output)
    output.mkdir(parents=True)
    selected = (
        limit if scope is not None else (len(ids) if maximum is None else min(maximum, len(ids)))
    )
    unselected = "not_run_scope" if scope is not None else "not_run_cap"
    rows = [
        dict(
            key=f"{match_id}__pt{i:04d}",
            clip=f"pt{i:04d}",
            status="pending" if n < selected else unselected,
            ledger_row=r,
        )
        for n, (i, r) in enumerate(zip(ids, ledger_rows, strict=True))
    ]
    snapshot = dict(
        schema="automatic_broadcast_shared_s6_v1",
        upstream_root=provenance.portable_path(upstream_root),
        match_id=match_id,
        observation_origin="automatic",
        execution_scope=execution_scope,
        policy=policy,
        policy_input=provenance.file_record(policy_path),
        backend_implementation=provenance.file_record(Path(__file__)),
        source_video=provenance.file_record(source_video),
        processing_video_path=str(source_video.resolve()),
        manifest=provenance.file_record(manifest_path),
        point_ledger=provenance.file_record(ledger),
        attempts=rows,
        runtime_frontier_model_calls=0,
        fitted_states_as_inputs=False,
        **(
            {
                "processing_scope": scope_module.describe(scope),
                "processing_scope_input": provenance.file_record(
                    scope_module.scope_path(upstream_root), role="automatic_execution_scope"
                ),
            }
            if scope is not None
            else {}
        ),
        **({"service_attempt_split": "on"} if service_attempt_split == "on" else {}),
        **({"observed_first_flight": "on"} if observed_first_flight == "on" else {}),
    )
    summary_path = output / "summary.json"

    def save():
        _save(summary_path, {**snapshot, "counts": summarize(rows, selected)})

    save()
    cache_started = time.monotonic()
    try:
        pts, pts_receipt = source_pts_cache(
            source_video, float(match["source_fps"]), upstream_root / "source_pts_cache"
        )
        snapshot["source_pts_preparation_seconds"] = time.monotonic() - cache_started
        snapshot["source_pts"] = provenance.file_record(pts)
        snapshot["source_pts_receipt"] = provenance.file_record(pts_receipt)
        receipts, parents = discover_upstream(upstream_root, match_id)
        if event_emissions is not None:
            event_receipt, event_parent = event_override_ancestry(
                event_emissions, event_provenance, source_video
            )
            receipts.append(event_receipt)
            parents.append(event_parent)
            snapshot["event_input_override"] = {
                "emissions": provenance.file_record(event_emissions),
                "producer": provenance.file_record(event_provenance),
            }
        if camera_projections is not None:
            camera_receipt, camera_parent = camera_override_ancestry(
                camera_projections, camera_provenance, source_video
            )
            receipts.append(camera_receipt)
            parents.append(camera_parent)
            snapshot["camera_input_override"] = {
                "projections": provenance.file_record(camera_projections),
                "producer": provenance.file_record(camera_provenance),
            }
        if player_observations is not None:
            player_receipt, player_parent = player_override_ancestry(
                player_observations, player_provenance, source_video
            )
            receipts.append(player_receipt)
            parents.append(player_parent)
            snapshot["player_input_override"] = {
                "observations": provenance.file_record(player_observations),
                "coordinates": provenance.file_record(
                    resolution.coordinate_manifest_path(player_observations)
                ),
                "producer": provenance.file_record(player_provenance),
            }
        if optional_contact_pose is not None:
            pose_receipt, pose_parent = optional_pose_ancestry(
                optional_contact_pose, optional_contact_pose_provenance, source_video
            )
            receipts.append(pose_receipt)
            parents.append(pose_parent)
            snapshot["optional_contact_pose_input"] = {
                "pose": provenance.file_record(optional_contact_pose),
                "automatic_view_intervals": provenance.file_record(directory / "shot_views_v1.csv"),
                "coordinates": provenance.file_record(
                    resolution.coordinate_manifest_path(optional_contact_pose)
                ),
                "producer": provenance.file_record(optional_contact_pose_provenance),
                "actor_players_replaced": False,
            }
    except Exception as error:
        for row in rows[:selected]:
            row.update(status="preparation_failed", reason=f"{type(error).__name__}: {error}")
        save()
        return summary_path

    def execute(index: int) -> tuple[int, dict]:
        row = dict(rows[index])
        point = output / "attempts" / row["key"]
        point.mkdir(parents=True)
        preparation_started = time.monotonic()
        try:
            frames = directory / "audit_frames_native_1080" / row["clip"]
            row["native_frames_directory"] = provenance.portable_path(frames)
            extraction = frames / "extraction_receipt.json"
            if extraction.is_file():
                row["extraction_receipt"] = provenance.file_record(extraction)
            inputs = AutomaticInputs(
                match_id=match_id,
                clip=row["clip"],
                manifest=manifest_path,
                point_ledger=ledger,
                extraction_receipt=frames / "extraction_receipt.json",
                frames_directory=frames,
                source_video=source_video,
                source_pts=pts,
                ball=directory / "ball_track_joint_native1080_arc_augmented_v2.csv",
                events=event_emissions or upstream_root / "event_emissions.json",
                cameras=camera_projections or directory / camera_name,
                players=players_path,
                ancestry=(),
                **(
                    {
                        "optional_contact_pose": optional_contact_pose,
                        "optional_contact_views": directory / "shot_views_v1.csv",
                    }
                    if optional_contact_pose is not None
                    else {}
                ),
            )
            ancestry_started = time.monotonic()
            parent = prepare_provenance(
                inputs=inputs,
                pts_receipt=pts_receipt,
                receipts=receipts,
                parents=parents,
                output=point / "pre_reconstruction.json",
            )
            row["pre_reconstruction_provenance"] = provenance.file_record(parent)
            row["pre_reconstruction_provenance_seconds"] = time.monotonic() - ancestry_started
            prepared = build_observations(
                replace(inputs, ancestry=(parent,)),
                point / "inputs",
                **(
                    {
                        "contact_prefix_scope": policy["contact_prefix_scope"],
                        "observation_partition": policy.get(
                            "observation_partition", "fifth_frame_withheld"
                        ),
                    }
                    if policy.get("contact_prefix_scope", "off") != "off"
                    else {}
                ),
                **(
                    {"optional_bounces": policy["optional_bounces"]}
                    if policy.get("optional_bounces", "off") != "off"
                    else {}
                ),
                **(
                    {"optional_contacts": "on"}
                    if policy.get("optional_contacts", "off") == "on"
                    else {}
                ),
                **(
                    {"optional_contact_composition": policy["optional_contact_composition"]}
                    if policy.get("optional_contact_composition", "off") != "off"
                    else {}
                ),
                **(
                    {"optional_contact_timing": policy["optional_contact_timing"]}
                    if policy.get("optional_contact_timing", "off") != "off"
                    else {}
                ),
                **(
                    {"optional_final_contacts": "on"}
                    if policy.get("optional_final_contacts", "off") == "on"
                    else {}
                ),
                **(
                    {"optional_interior_contacts": "on"}
                    if policy.get("optional_interior_contacts", "off") == "on"
                    else {}
                ),
                **(
                    {"observed_horizon_tail": "on"}
                    if policy.get("observed_horizon_tail", "off") == "on"
                    else {}
                ),
                # Its sibling in `DROPPED_TAIL_KEYS`, forwarded the same way. Until
                # 2026-09-18 this key reached the shared stage's FIT and never the
                # automatic PREPARATION, so an automatic policy could model a terminal
                # net tail on a scope that had refused to admit one.
                **(
                    {"terminal_net_tail": "on"}
                    if policy.get("terminal_net_tail", "off") == "on"
                    else {}
                ),
                # Component admission must reach ordinary video inference as
                # well as supplied-input preparation. Preserve absent defaults.
                **(
                    {
                        "contact_components": policy.get("contact_components", "off"),
                        **{
                            name: policy[name]
                            for name in COMPONENT_FORWARDED
                            if policy.get(name, "off") != "off"
                        },
                    }
                    if any(policy.get(name, "off") != "off" for name in COMPONENT_TRIGGERS)
                    else {}
                ),
                # Default off; forwarded on its own because it is a ball-row filter,
                # not an admission, and must not drag `contact_components` with it.
                **(
                    {"automatic_ball_jump_rejection": policy["automatic_ball_jump_rejection"]}
                    if policy.get("automatic_ball_jump_rejection", "off") != "off"
                    else {}
                ),
                **(
                    {"automatic_ball_streak_centre": policy["automatic_ball_streak_centre"]}
                    if policy.get("automatic_ball_streak_centre", "off") != "off"
                    else {}
                ),
                # Production default is admit inside build_observations. An explicit
                # policy value, including the rollback "off", replaces that default.
                **(
                    {"intrinsic_fallback_registration": policy["intrinsic_fallback_registration"]}
                    if "intrinsic_fallback_registration" in policy
                    else {}
                ),
                **(
                    {"near_baseline_refinement": policy["near_baseline_refinement"]}
                    if policy.get("near_baseline_refinement", "off") != "off"
                    else {}
                ),
                **(
                    {"shot_homography_propagation": policy["shot_homography_propagation"]}
                    if policy.get("shot_homography_propagation", "off") != "off"
                    else {}
                ),
                **(
                    {"player_body_scale": policy["player_body_scale"]}
                    if policy.get("player_body_scale", "off") != "off"
                    else {}
                ),
                **({"service_attempt_split": "on"} if service_attempt_split == "on" else {}),
                **({"observed_first_flight": "on"} if observed_first_flight == "on" else {}),
            )
            row["preparation_seconds"] = time.monotonic() - preparation_started
            row["preparation"] = provenance.file_record(point / "inputs" / "preparation.json")
            row["evidence"] = prepared["evidence"]
            row["observation_row"] = prepared["row"]
            if prepared["status"] == "prepared_children":
                children = []
                row["children"] = children
                for n, child in enumerate(prepared["children"], 1):
                    child_point = point / f"child_{n:02d}"
                    child_point.mkdir()
                    child_row = dict(
                        key=f"{row['key']}__a{n:02d}",
                        parent_key=row["key"],
                        ledger_row=row["ledger_row"],
                        preparation=provenance.file_record(
                            point / "inputs" / f"child_{n:02d}" / "preparation.json"
                        ),
                        extraction_receipt=row.get("extraction_receipt"),
                        native_frames_directory=row.get("native_frames_directory"),
                        clip=row["clip"],
                        child_index=n,
                        evidence=child["evidence"],
                        observation_row=child["row"],
                        service_attempt_scope=child.get("service_attempt_scope"),
                    )
                    if child["status"] != "prepared":
                        child_row.update(status="preparation_held", reason=child.get("reason"))
                    else:
                        job = child_point / "job.json"
                        _save(job, {"row": child["row"], "policy": policy})
                        child_row.update(run_stage(job, child_point / "stage", timeout_seconds))
                    children.append(child_row)
                row.update(
                    children=children,
                    original_parent_retained=True,
                    status="completed"
                    if all(c["status"] == "completed" for c in children)
                    else "children_incomplete",
                    verdict={
                        "complete_point": all(
                            c.get("verdict", {}).get("complete_point", False) for c in children
                        ),
                        "accepted_flight_count": sum(
                            c.get("verdict", {}).get("accepted_flight_count", 0) for c in children
                        ),
                        "flight_count": sum(
                            c.get("verdict", {}).get("flight_count", 0) for c in children
                        ),
                        "semantics": "all child legacy gates required; censored aftermath and unknown ending remain explicit",
                    },
                )
            elif prepared["status"] != "prepared":
                row.update(status="preparation_held", reason=prepared.get("reason"))
            else:
                job = point / "job.json"
                _save(job, {"row": prepared["row"], "policy": policy})
                row.update(run_stage(job, point / "stage", timeout_seconds))
        except Exception as error:
            row.update(status="preparation_failed", reason=f"{type(error).__name__}: {error}")
        return index, row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(execute, i) for i in range(selected)]
        for future in as_completed(pending):
            index, row = future.result()
            rows[index] = row
            save()
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-points", type=int)
    parser.add_argument(
        "--event-emissions", type=Path, help="optional uniformly qualified automatic event artifact"
    )
    parser.add_argument(
        "--event-provenance",
        type=Path,
        help="automatic producer binding the qualification stage receipt",
    )
    parser.add_argument("--player-observations", type=Path)
    parser.add_argument("--player-provenance", type=Path)
    parser.add_argument("--service-attempt-split", choices=("off", "on"), default="off")
    parser.add_argument("--observed-first-flight", choices=("off", "on"), default="off")
    parser.add_argument("--camera-projections", type=Path)
    parser.add_argument("--camera-provenance", type=Path)
    parser.add_argument("--optional-contact-pose", type=Path)
    parser.add_argument("--optional-contact-pose-provenance", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--attempt-timeout-seconds", type=float, default=4200)
    args = parser.parse_args()
    print(
        run_cached(
            upstream_root=args.upstream_root,
            match_id=args.match_id,
            source_video=args.source_video,
            policy_path=args.policy,
            output=args.output,
            maximum=args.max_points,
            workers=args.workers,
            timeout_seconds=args.attempt_timeout_seconds,
            event_emissions=args.event_emissions,
            event_provenance=args.event_provenance,
            player_observations=args.player_observations,
            player_provenance=args.player_provenance,
            service_attempt_split=args.service_attempt_split,
            observed_first_flight=args.observed_first_flight,
            camera_projections=args.camera_projections,
            camera_provenance=args.camera_provenance,
            optional_contact_pose=args.optional_contact_pose,
            optional_contact_pose_provenance=args.optional_contact_pose_provenance,
        )
    )


if __name__ == "__main__":
    main()
