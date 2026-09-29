"""Authoritative label-free raw-broadcast runner.

The point ledger is intentionally abstaining and remains scientifically experimental. The runner
exists so every raw-video result follows one provenance-carrying implementation rather than a set
of match-specific shell scripts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from cv.pipeline.artifact_cache import (
    receipt_path,
    stage_receipt_allows_dependency_narrowing,
    stage_receipt_matches,
    write_stage_receipt,
)
from cv.pipeline.automatic_ball_track import (
    CURRENT_EVENT_TRACK,
    LEGACY_EVENT_TRACK,
    current_track_inputs,
    validate_event_track,
)
from cv.pipeline.broadcast_source import probe, processing_source
from cv.pipeline.camera_artifacts import slice_reconstruction_match_artifacts
from cv.pipeline.canonical_runner import POINT_MAP_NAME, REEL_NAME, resolve_court_jobs, run_manifest
from cv.pipeline.court_topology import (
    REGISTRATION_MASK_POLICIES,
    INTERPOLATED_CAMERA_POLICIES,
    SURFACE_WITNESS_POLICIES,
    resolve_surface_witness_policy,
)
from cv.pipeline import evaluation_scope
from cv.pipeline.paths import data_root, processed_root, tracker_root
from cv.pipeline.point_ledger import POINT_CONTINUATION_MAX_GAP_SECONDS
from cv.pipeline.provenance import (
    AUTOMATIC_MODE,
    build_provenance,
    file_record,
    load_provenance,
    write_provenance,
)
from cv.pipeline.vlm_policy import configured_vlm_model
from physics import bounce_reference
from cv.pipeline.tracking_composition import (
    ANCHOR_EXTENSION_REPORT_NAME,
    OWNERSHIP_ARTIFACT_NAME,
    run_match as run_tracking_match,
    tracking_model_inputs,
)

PIPELINE = Path(__file__).resolve().parent
REPO = PIPELINE.parents[1]
DEFAULT_GATES = PIPELINE / "default_gates.json"
POINT_LEDGER_STOP = "point-ledger"
EVENT_INPUTS_STOP = "event-inputs"
#: Every automatic observation built (through contact strikers); no reconstruction.
UPSTREAM_STOP = "upstream"
# The shipped Stage-1 heads.  `docs/wk1/s1_precision.md` records that the Roland Garros run of
# record was launched against `processed/wk2_s1`, the pass-1 arm, because a missing shipped
# directory used to fall back silently; the runner now fails closed instead, and any other arm
# has to be named on the command line.
SHIPPED_SEGMENTATION_MODELS = "models/pipeline/s1_stage_v1"
VIEW_HEAD_NAME = "view_head_v1.json"
SERVE_HEAD_NAME = "serve_heads_v1.pkl"


class NoProcessableAttemptsError(RuntimeError):
    """The automatic point ledger abstained on every candidate interval."""


class MissingSegmentationModelsError(RuntimeError):
    """The serve Stage-1 heads the runner defaults to are not installed."""


def shipped_segmentation_models() -> Path:
    return data_root() / SHIPPED_SEGMENTATION_MODELS


def resolve_segmentation_models(directory: Path) -> tuple[Path, Path]:
    """The view head and serve head under ``directory``, or a failure naming what is missing.

    Fails closed: a Stage-1 run with no heads is not silently downgraded to the legacy
    audio-gap assembler, because a ledger built by a different assembler than the one asked
    for is indistinguishable in its own artifacts from the one that was asked for.
    """
    view_head = Path(directory) / VIEW_HEAD_NAME
    serve_head = Path(directory) / SERVE_HEAD_NAME
    missing = [path for path in (view_head, serve_head) if not path.is_file()]
    if missing:
        raise MissingSegmentationModelsError(
            "serve segmentation models missing: "
            + ", ".join(os.fspath(path) for path in missing)
            + f"; the shipped Stage-1 heads belong under {shipped_segmentation_models()}. "
            "Pass --segmentation-models to run another arm (the pass-1 heads under "
            "processed/wk2_s1 are the Roland Garros run of record), or --segmentation legacy "
            "for the audio-gap ledger."
        )
    return view_head, serve_head


class Runner:
    def __init__(self, out_root: Path, resume: bool) -> None:
        self.out_root = out_root
        self.resume = resume
        self.receipts: dict[str, Path] = {}

    @property
    def upstream_receipts(self) -> list[Path]:
        return list(self.receipts.values())

    def stage(
        self,
        name: str,
        command: list[str],
        inputs: list[Path],
        outputs: list[Path],
        callback=None,
        dependencies: tuple[str, ...] = (),
        *,
        cached_inputs=None,
    ) -> None:
        """Run, reuse or migrate one stage.

        ``cached_inputs`` is an optional predicate over the inputs a *previous* run of
        this stage actually consumed but the stage identity cannot name -- files chosen
        during the run, and evidence that was absent.  It is consulted only when a
        receipt would otherwise be honoured, and it vetoes both reuse and dependency
        narrowing: an identical command over identical declared inputs is still stale if
        what it read underneath has changed.
        """
        upstream_receipts = [self.receipts[dependency] for dependency in dependencies]
        reusable = self.resume and stage_receipt_matches(
            out_dir=self.out_root,
            stage=name,
            command=command,
            inputs=inputs,
            outputs=outputs,
            upstream_receipts=upstream_receipts,
        )
        migrated = (
            self.resume
            and not reusable
            and stage_receipt_allows_dependency_narrowing(
                out_dir=self.out_root,
                stage=name,
                command=command,
                inputs=inputs,
                outputs=outputs,
                upstream_receipts=upstream_receipts,
            )
        )
        if (reusable or migrated) and cached_inputs is not None and not cached_inputs():
            reusable = migrated = False
        action = "reuse" if reusable else "migrate" if migrated else "run"
        print(f"[{action}] {name}", flush=True)
        if migrated:
            write_stage_receipt(
                out_dir=self.out_root,
                stage=name,
                command=command,
                inputs=inputs,
                outputs=outputs,
                upstream_receipts=upstream_receipts,
            )
        if not reusable and not migrated:
            if callback is None:
                subprocess.run(command, cwd=REPO, check=True)
            else:
                callback()
            write_stage_receipt(
                out_dir=self.out_root,
                stage=name,
                command=command,
                inputs=inputs,
                outputs=outputs,
                upstream_receipts=upstream_receipts,
            )
        self.receipts[name] = receipt_path(self.out_root, name)


def _run_many(commands: list[list[str]]) -> None:
    for command in commands:
        subprocess.run(command, cwd=REPO, check=True)


def _write_selected_point_map(ledger: Path, destination: Path, selected: int) -> None:
    """Copy the first ``selected`` ledger rows verbatim, header and columns unchanged.

    Every declared native window, PTS, score and continuation column of a selected row is
    carried over; nothing is shortened, renumbered or reordered.
    """
    with ledger.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)[:selected]
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def materialize_postseg(
    *,
    video: Path,
    match_out: Path,
    ledger: Path,
    manifest_path: Path,
    match_id: str,
    fps: float,
    surface: str,
    scope: dict | None = None,
) -> dict:
    """Materialize the post-segmentation work roster.

    Without a declared scope this is unchanged: the whole ledger is the work map and one
    manifest plays both roles.  With a scope the two roles separate -- ``broadcast_manifest``
    still names the complete ledger roster, for S6 identity and accounting, while the work
    map, ``processing_manifest.json`` and the conventional ``manifest.json`` name only the
    declared prefix that the per-point stages will actually produce.
    """
    point_rows = list(csv.DictReader(ledger.open()))
    if not point_rows:
        raise NoProcessableAttemptsError("automatic point ledger emitted no processable attempts")
    reel = match_out / REEL_NAME
    if reel.exists() or reel.is_symlink():
        reel.unlink()
    reel.symlink_to(video.resolve())
    if scope is None:
        shutil.copyfile(ledger, match_out / POINT_MAP_NAME)
    else:
        _write_selected_point_map(ledger, match_out / POINT_MAP_NAME, scope["selected_attempts"])
    manifest = {
        "schema": "broadcast_pipeline_manifest_v1",
        "points_per_match": len(point_rows),
        "matches": [
            {
                "id": match_id,
                "source_fps": fps,
                "surface": surface,
                "point_ids": [int(row["pt"]) for row in point_rows],
            }
        ],
    }
    encoded_manifest = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(encoded_manifest)
    # point_grammar is shared with canonical inference and deliberately expects
    # this conventional filename at the processed root.  Keep the explicitly
    # named runner manifest too, but materialize the compatible automatic copy.
    root = manifest_path.parent
    if scope is None:
        (root / evaluation_scope.CONVENTIONAL_MANIFEST_NAME).write_text(encoded_manifest)
        return manifest
    work = evaluation_scope.work_manifest(manifest, list(scope["selected_point_ids"]))
    encoded_work = json.dumps(work, indent=2, sort_keys=True) + "\n"
    (root / evaluation_scope.WORK_MANIFEST_NAME).write_text(encoded_work)
    # `event_video_model --root` and `point_grammar.annotate_from_root` read this
    # conventional name, so the selected roster has to be the copy that lands there.
    (root / evaluation_scope.CONVENTIONAL_MANIFEST_NAME).write_text(encoded_work)
    evaluation_scope.write_scope(evaluation_scope.scope_path(root), scope)
    return work


def reconstruction_point_arguments(ledger: Path, match_id: str, maximum: int | None) -> list[str]:
    """Restrict expensive S6 work without changing upstream point accounting."""
    if maximum is None:
        return []
    if maximum < 1:
        raise ValueError("--max-points must be at least one")
    rows = list(csv.DictReader(ledger.open()))
    # A cap that includes the entire ledger does not need a selector.  Besides
    # keeping the command simpler, this lets reconstruction discover its
    # authoritative automatic point universe itself.
    if maximum >= len(rows):
        return []
    rows = rows[:maximum]
    return [
        value
        for row in rows
        # Reconstruction's automatic point universe uses globally-qualified
        # keys even though its materialized clip directories use local names.
        for value in ("--point", f"{match_id}__pt{int(row['pt']):04d}")
    ]


def tracking_command(
    python: str,
    manifest: Path,
    out: Path,
    device: int,
    *,
    optimized_local_runtime: bool = True,
    primary_ball_ownership: bool = False,
    anchor_extension: bool = False,
    guide_sequence_association: bool = False,
) -> list[str]:
    """The declared tracking stage command.

    The optional ball-identity arms are named here because the stage receipt is built from
    this list: an arm that only reached the in-process call would leave the fingerprint
    unchanged and a resumed run would keep the other arm's ball track. The sequence-association
    arm writes no side artifact of its own, so the declared flag is the only thing that
    separates its composed track from the default one.
    """
    return [
        python,
        os.fspath(PIPELINE / "tracking_composition.py"),
        "--manifest",
        os.fspath(manifest),
        "--out",
        os.fspath(out),
        "--device",
        str(device),
        "--optimized-full-runtime",
        *(["--optimized-local-runtime"] if optimized_local_runtime else []),
        *(["--primary-ball-ownership"] if primary_ball_ownership else []),
        *(["--anchor-extension"] if anchor_extension else []),
        *(["--guide-sequence-association"] if guide_sequence_association else []),
    ]


def tracking_outputs(
    match_out: Path,
    *,
    primary_ball_ownership: bool = False,
    anchor_extension: bool = False,
) -> list[Path]:
    """Consumed ball track, plus the side artifacts each optional arm must have written."""
    return [
        match_out / "ball_track_joint_native1080_arc_augmented_v2.csv",
        *([match_out / OWNERSHIP_ARTIFACT_NAME] if primary_ball_ownership else []),
        *([match_out / ANCHOR_EXTENSION_REPORT_NAME] if anchor_extension else []),
    ]


def point_gate_commands(
    python: str,
    out: Path,
    manifest: Path,
    *,
    court_loss_policy: str = "strict",
    native_actor_continuity: bool = False,
    native_cut_continuity: bool = False,
    retained_play_scope: bool = False,
    court_anchor_camera: bool = False,
) -> list[list[str]]:
    """Return the runner gate chain in the same order as the cohort chain."""
    active = out / "active_play_v1.json"
    cadence = out / "frame_cadence_audit_v1.json"
    court = out / "court_geometry_point_gate_v1.json"
    validity = out / "point_validity_gate_v1.json"
    tracking = out / "untouched_tracking_point_gate_v1.json"
    camera_support = out / "reliable_per_frame_camera_support_v1.json"
    return [
        [
            python,
            "-m",
            "cv.pipeline.active_play_gate",
            "--processed",
            os.fspath(out),
            "--out",
            os.fspath(active),
            "--court-loss-policy",
            court_loss_policy,
            *(["--native-actor-continuity"] if native_actor_continuity else []),
            *(["--native-cut-continuity"] if native_cut_continuity else []),
            *(["--court-anchor-camera"] if court_anchor_camera else []),
        ],
        [
            python,
            "-m",
            "cv.pipeline.tracking_point_gate",
            "--manifest",
            os.fspath(manifest),
            "--protocol",
            os.fspath(DEFAULT_GATES),
            "--out",
            os.fspath(out),
            "--active-play",
            os.fspath(active),
        ],
        [
            python,
            "-m",
            "cv.pipeline.frame_cadence",
            "--audit-root",
            os.fspath(out),
            "--active-play",
            os.fspath(active),
            "--out",
            os.fspath(cadence),
        ],
        [
            python,
            "-m",
            "cv.pipeline.court_geometry_point_gate",
            "--manifest",
            os.fspath(manifest),
            "--out",
            os.fspath(out),
            "--output",
            os.fspath(court),
        ],
        *(
            [
                [
                    python,
                    "-m",
                    "cv.pipeline.camera_frame_support",
                    "--manifest",
                    os.fspath(manifest),
                    "--out",
                    os.fspath(out),
                    "--output",
                    os.fspath(camera_support),
                ]
            ]
            if retained_play_scope
            else []
        ),
        [
            python,
            "-m",
            "cv.pipeline.compose_point_validity_gate",
            "--active-play",
            os.fspath(active),
            "--tracking-gate",
            os.fspath(tracking),
            "--court-geometry",
            os.fspath(court),
            "--frame-cadence",
            os.fspath(cadence),
            "--maximum-invalid-fraction",
            "absolute",
            *(
                ["--retained-play-scope", "--camera-frame-support", os.fspath(camera_support)]
                if retained_play_scope
                else []
            ),
            "--output",
            os.fspath(validity),
        ],
    ]


def validate_event_marginal_threshold(value: float | None) -> None:
    """Validate an explicitly selected event operating threshold before expensive work."""
    if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
        raise ValueError("--event-marginal-threshold must be a finite probability in [0, 1]")


def event_inference_command(
    python: str,
    out: Path,
    checkpoint: Path,
    *,
    device: int = 0,
    supported_impulse_events: bool = False,
    native_streak_fallback: bool = False,
    model_accepted_tracking_held_events: bool = False,
    native_net_evidence: bool = False,
    native_ground_evidence: bool = False,
    event_court_geometry: str = "point_static",
    event_court_frame_missing: str = "hold",
    event_track: str = LEGACY_EVENT_TRACK,
    marginal_threshold: float | None = None,
    live_shot_camera: bool = False,
) -> list[str]:
    validate_event_track(event_track)
    validate_event_marginal_threshold(marginal_threshold)
    return [
        python,
        "-m",
        "cv.pipeline.event_video_model",
        "--device",
        f"cuda:{device}",
        "--root",
        os.fspath(out),
        "--checkpoint",
        os.fspath(checkpoint),
        "--crops",
        os.fspath(out / "event_crops_v1"),
        "--tracking-gate",
        os.fspath(out / "untouched_tracking_point_gate_v1.json"),
        "--point-gate",
        os.fspath(out / "point_validity_gate_v1.json"),
        "--output",
        os.fspath(out / "event_emissions.json"),
        "--jobs",
        "8",
        "--workers",
        "8",
        *(["--supported-impulse-events"] if supported_impulse_events else []),
        *(["--native-streak-fallback"] if native_streak_fallback else []),
        *(["--model-accepted-tracking-held-events"] if model_accepted_tracking_held_events else []),
        *(["--native-net-evidence"] if native_net_evidence else []),
        *(["--native-ground-evidence"] if native_ground_evidence else []),
        "--court-geometry",
        event_court_geometry,
        "--court-frame-missing",
        event_court_frame_missing,
        # A default run's command is unchanged; a selection declares itself, so
        # the stage fingerprint separates the two arms.
        *(["--event-track", event_track] if event_track != LEGACY_EVENT_TRACK else []),
        *(
            ["--operating-point", "none", "--marginal-threshold", str(marginal_threshold)]
            if marginal_threshold is not None
            else []
        ),
        *(["--live-shot-camera"] if live_shot_camera else []),
    ]


def event_evidence_inputs(
    out: Path,
    match_id: str,
    *,
    native_net_evidence: bool,
    native_ground_evidence: bool,
    native_streak_fallback: bool = False,
    event_track: str = LEGACY_EVENT_TRACK,
) -> list[Path]:
    """Bind exactly the optional evidence sources, including guide ancestry/sidecars.

    A non-legacy ``event_track`` also binds the selected ball track, its
    coordinate sidecar, the guide CSV that sidecar declares and the guide's own
    sidecar, so a byte change anywhere in the bound ancestry -- or a change of
    selection -- re-runs event inference instead of resuming on stale emissions.
    """
    validate_event_track(event_track)
    snapshots = []
    if native_ground_evidence:
        from cv.pipeline.event_ground_evidence import GroundSnapshot

        snapshots.append(GroundSnapshot.load(out, [match_id]))
    if native_net_evidence:
        from cv.pipeline.event_net_evidence import NetSnapshot

        snapshots.append(NetSnapshot.load(out, [match_id]))
    selected = (
        current_track_inputs(Path(out) / match_id) if event_track != LEGACY_EVENT_TRACK else []
    )
    if native_streak_fallback:
        # The cadence audit, the native coordinate space and the per-clip extraction
        # receipts decide which original pictures the streak measurement may read at all
        # and what bytes they must have, so a byte change in any of them re-runs event
        # inference rather than resuming on stale emissions.  The receipts are a few
        # hundred small documents; the pictures themselves are not hashed here.
        from cv.pipeline.event_native_pictures import (
            CADENCE_AUDIT,
            CADENCE_SIDECAR,
            EXTRACTION_RECEIPT,
            FRAMES_DIR,
        )

        declared = [Path(out) / CADENCE_AUDIT, Path(out) / match_id / CADENCE_SIDECAR]
        # Absent ancestry is absent evidence for the measurement and must not fail the
        # identity build; a file that later appears is caught by the cached-input check.
        selected = [
            *selected,
            *(path for path in declared if path.is_file()),
            *sorted((Path(out) / match_id / FRAMES_DIR).glob(f"*/{EXTRACTION_RECEIPT}")),
        ]
    return sorted({path for snapshot in snapshots for path in snapshot.track.paths} | set(selected))


def consumed_event_inputs_unchanged(manifest: Path, support: Path | None = None) -> bool:
    """Do the inputs the previous event run actually consumed still say the same thing?

    Which original pictures an optional evidence stage reads is decided during the run,
    not at graph-build time, so the stage identity cannot name them.  The previous run
    already recorded them: every consulted file is a ``provenance.file_record`` in the
    emission manifest's ``impulse_event_support.inputs``, and every path that was looked
    for and was *absent* is named in the support artifact.  This re-checks exactly those
    two lists -- no directory is walked and no unconsulted picture is hashed.

    Anything unreadable, unresolvable or changed answers ``False``: a run whose prior
    consumption cannot be verified is re-run, never resumed.
    """
    from cv.pipeline import provenance
    from cv.pipeline.s6_automatic_observations import _resolve

    try:
        document = json.loads(Path(manifest).read_text())
        records = document["impulse_event_support"]["inputs"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    try:
        for record in records:
            path = _resolve(record)
            if provenance.file_record(path, role=record.get("role")) != record:
                return False
    except (OSError, ValueError, TypeError):
        return False
    if support is None:
        return True
    try:
        expected = document["native_streak_fallback"]["artifact"]
        if provenance.file_record(support, role="native_streak_fallback_support") != expected:
            return False
        absent = json.loads(Path(support).read_text())["expected_absent_inputs"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    try:
        # Evidence that has since arrived is new evidence: an unavailable-support hold
        # taken while it was missing is not a result this resume may keep.
        return not any(_resolve(record).exists() for record in absent)
    except (OSError, ValueError, TypeError):
        return False


def nightly_reconstruction_arguments(out: Path) -> list[str]:
    """Arguments shared with cv.validation.wk3_nightly.reconstruct."""
    return [
        "--include-dead-time-emissions",
        "--max-nfev",
        "20",
        "--workers",
        "8",
        "--math-threads",
        "1",
        "--point-timeout-seconds",
        "600",
        "--terminal-flights",
        "--anchor-bounce-geometry",
        "--whole-point-branches",
        "--branch-width",
        "3",
        "--branch-margin",
        "8",
        "--anchors-output-root",
        os.fspath(out / "reconstruction_anchors"),
    ]


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _event_matches_point(row: dict, match_id: str, clip: str) -> bool:
    value = str(row.get("clip", ""))
    return (
        value in {clip, f"{match_id}__{clip}", f"{match_id}/{clip}"}
        and str(row.get("match_id", match_id)) == match_id
    )


def prepare_reconstruction_point_inputs(
    *,
    out: Path,
    manifest_path: Path,
    emissions: Path,
    destination: Path,
    point_keys: set[str] | None = None,
    pose_artifact_name: str | None = None,
    physical_motion_name: str | None = None,
    camera_artifact_name: str = "camera_P_per_frame_v1.npz",
) -> Path:
    """Write one cohort-shaped reconstruction root per automatic point."""
    manifest = json.loads(manifest_path.read_text())
    active_paths = sorted(
        path
        for path in out.glob("active_play_v*.json")
        if not path.name.endswith(".provenance.json")
    )
    if not active_paths:
        raise FileNotFoundError(f"active-play artifact is absent under {out}")
    active = json.loads(active_paths[-1].read_text())
    tracking = json.loads((out / "untouched_tracking_point_gate_v1.json").read_text())
    validity_path = out / "point_validity_gate_v1.json"
    validity = json.loads(validity_path.read_text()) if validity_path.is_file() else None
    emission_payload = json.loads(emissions.read_text())
    emission_rows = (
        emission_payload.get("emissions", [])
        if isinstance(emission_payload, dict)
        else emission_payload
    )

    roots: list[dict] = []
    slice_reports: list[dict] = []
    for match in manifest["matches"]:
        match_id = str(match["id"])
        clips = sorted(
            key.split("/", 1)[1]
            for key in active
            if key.startswith(f"{match_id}/")
            and (point_keys is None or key.replace("/", "__", 1) in point_keys)
        )
        if not clips:
            continue
        point_match_dirs = {clip: destination / match_id / clip / match_id for clip in clips}
        slice_report = slice_reconstruction_match_artifacts(
            out / match_id,
            point_match_dirs,
            camera_artifact_name=camera_artifact_name,
            pose_artifact_name=pose_artifact_name,
            physical_motion_name=physical_motion_name,
        )
        tracking_rows = {
            f"{row['match_id']}__{row['clip']}": row for row in tracking.get("rows", [])
        }
        validity_rows = (
            {f"{row['match_id']}__{row['clip']}": row for row in validity.get("rows", [])}
            if validity is not None
            else {}
        )
        for clip in clips:
            point_key = f"{match_id}__{clip}"
            active_key = f"{match_id}/{clip}"
            point_root = destination / match_id / clip
            point_manifest = {
                **manifest,
                "points_per_match": 1,
                "matches": [{**match, "point_ids": [int(clip.removeprefix("pt"))]}],
            }
            _write_json(point_root / "manifest.json", point_manifest)
            _write_json(point_root / active_paths[-1].name, {active_key: active[active_key]})
            _write_json(
                point_root / "untouched_tracking_point_gate_v1.json",
                {"rows": [tracking_rows[point_key]]},
            )
            if validity is not None:
                _write_json(
                    point_root / validity_path.name,
                    {"rows": [validity_rows[point_key]]},
                )
            point_events = [
                row for row in emission_rows if _event_matches_point(row, match_id, clip)
            ]
            point_emissions = (
                {**emission_payload, "emissions": point_events}
                if isinstance(emission_payload, dict)
                else point_events
            )
            _write_json(point_root / emissions.name, point_emissions)
            roots.append(
                {
                    "point": point_key,
                    "match_id": match_id,
                    "clip": clip,
                    "root": os.fspath(point_root),
                    "manifest": os.fspath(point_root / "manifest.json"),
                    "event_boundaries": os.fspath(point_root / emissions.name),
                }
            )
        slice_report["point_roots"] = len(clips)
        slice_reports.append(slice_report)

    layout = {
        "schema": "reconstruction_point_input_layout_v1",
        "source_root": os.fspath(out),
        "source_manifest": os.fspath(manifest_path),
        "camera_artifact_name": camera_artifact_name,
        "pose_artifact_name": pose_artifact_name,
        "physical_motion_name": physical_motion_name,
        "slice_reports": slice_reports,
        "points": roots,
    }
    layout_path = destination / "manifest.json"
    _write_json(layout_path, layout)
    return layout_path


def run_point_reconstruction_layout(
    *,
    layout_path: Path,
    output: Path,
    anchors_output_root: Path,
    reports_root: Path,
    timings_output: Path,
    workers: int = 8,
) -> dict:
    """Run point-local nightly commands concurrently and aggregate their reports."""
    if workers != 8:
        raise ValueError("the reconstruction layout is measured with eight workers")
    layout = json.loads(layout_path.read_text())
    points = list(layout["points"])
    reports_root.mkdir(parents=True, exist_ok=True)
    logs_root = reports_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)

    def evaluate(row: dict) -> dict:
        point = str(row["point"])
        report_path = reports_root / str(row["match_id"]) / f"{row['clip']}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = logs_root / f"{point}.log"
        command = [
            sys.executable,
            os.fspath(PIPELINE / "reconstruct_3d.py"),
            "--audit-root",
            str(row["root"]),
            "--manifest",
            str(row["manifest"]),
            "--event-boundaries",
            str(row["event_boundaries"]),
            "--output",
            os.fspath(report_path),
            *nightly_reconstruction_arguments(anchors_output_root.parent),
        ]
        # ``nightly_reconstruction_arguments`` derives this path from its root;
        # make the explicit shared destination authoritative for point aggregation.
        anchor_index = command.index("--anchors-output-root") + 1
        command[anchor_index] = os.fspath(anchors_output_root)
        started = time.perf_counter()
        with log_path.open("w") as log:
            completed = subprocess.run(
                command,
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed = time.perf_counter() - started
        return {
            "point": point,
            "seconds": elapsed,
            "returncode": completed.returncode,
            "report": os.fspath(report_path),
            "log": os.fspath(log_path),
            "command": command,
        }

    started = time.perf_counter()
    timing_rows: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(evaluate, row) for row in points]
        for future in concurrent.futures.as_completed(futures):
            timing_rows.append(future.result())
    wall_seconds = time.perf_counter() - started
    timing_rows.sort(key=lambda row: row["point"])
    timing_payload = {
        "schema": "reconstruction_point_timing_v1",
        "workers": workers,
        "points": len(timing_rows),
        "wall_seconds": wall_seconds,
        "aggregate_wall_seconds_per_point": wall_seconds / max(len(timing_rows), 1),
        "median_point_process_seconds": (
            sorted(row["seconds"] for row in timing_rows)[len(timing_rows) // 2]
            if timing_rows
            else None
        ),
        "rows": timing_rows,
    }
    _write_json(timings_output, timing_payload)
    failed = [row for row in timing_rows if row["returncode"]]
    if failed:
        raise RuntimeError(f"{len(failed)} point reconstruction commands failed: {failed[:3]}")

    details = []
    for row in timing_rows:
        report = json.loads(Path(row["report"]).read_text())
        point_details = report.get("points_detail", [])
        if len(point_details) != 1 or point_details[0].get("point") != row["point"]:
            raise ValueError(f"point report identity mismatch: {row['report']}")
        details.append(point_details[0])
    from cv.pipeline.reconstruction import summarize

    report = summarize(details, owner_truth_loaded=False)
    report["match_shared_priors"] = None
    report["execution"] = {
        "point_workers": workers,
        "math_threads_per_worker": 1,
        "point_timeout_seconds": 600.0,
        "point_local_input_layout": True,
        "point_input_layout": os.fspath(layout_path),
        "point_timing": os.fspath(timings_output),
    }
    _write_json(output, report)
    return report


def ledger_point_summary(ledger: Path) -> dict:
    """How many attempts the ledger holds and how many score points they group into."""
    rows = list(csv.DictReader(ledger.open()))
    points = [row["point_index"] for row in rows if row.get("point_index") not in (None, "")]
    multiple = {
        row["point_index"] for row in rows if row.get("attempts_in_point") not in (None, "", "1")
    }
    return {
        "attempts": len(rows),
        "score_points": len(set(points)) if points else None,
        "continuation_attempts": sum(
            1 for row in rows if row.get("attempt_role") == "continuation"
        ),
        "multi_attempt_points": len(multiple) if points else None,
    }


def write_event_inputs_manifest(
    *,
    args: argparse.Namespace,
    runner: Runner,
    processing_video: Path,
    ledger: Path,
    work_manifest: Path,
) -> Path:
    """Publish the ordinary upstream boundary without running or claiming S5/S6."""
    match_out = args.out / args.match_id
    required = {"point_ledger", "postseg_base", "tracking", "point_gates"}
    if independent_contact_pose_requested(args):
        required.add("pose_player_crop")
    missing = required - runner.receipts.keys()
    if missing:
        raise ValueError(f"event-input boundary lacks completed stages: {sorted(missing)}")
    outputs = [
        file_record(ledger, role="automatic_point_ledger"),
        file_record(work_manifest, role="automatic_work_manifest"),
    ]
    marker = evaluation_scope.scope_path(args.out)
    if marker.is_file():
        outputs.append(file_record(marker, role="automatic_evaluation_scope"))
    if independent_contact_pose_requested(args):
        outputs.extend(
            [
                file_record(args.shared_s6_optional_contact_pose, role="automatic_contact_pose"),
                file_record(
                    args.shared_s6_optional_contact_pose_provenance,
                    role="automatic_contact_pose_producer",
                ),
            ]
        )
    document = {
        "schema": "broadcast_event_inputs_run_v1",
        "provenance": build_provenance(
            root=REPO,
            mode=AUTOMATIC_MODE,
            source_videos=[file_record(processing_video, role="source_broadcast")],
            configuration={
                "runner": "cv.pipeline.broadcast_runner",
                "stop_after": EVENT_INPUTS_STOP,
                "match_id": args.match_id,
                "surface": args.surface,
                "evaluation_first_attempts": requested_evaluation_scope(args),
                # Declared only when selected, so a default boundary manifest is unchanged.
                **(
                    {"active_play_native_cut_continuity": True}
                    if getattr(args, "active_play_native_cut_continuity", False)
                    else {}
                ),
                **(
                    {"retained_play_event_gate": True}
                    if getattr(args, "retained_play_event_gate", False)
                    else {}
                ),
                "active_play_court_anchor_camera": getattr(
                    args, "active_play_court_anchor_camera", True
                ),
                "stages": list(runner.receipts),
                "event_inference_executed": False,
                "reconstruction_executed": False,
            },
            reused_artifacts=[
                file_record(args.out / "provenance.json", role="automatic_parent_provenance"),
                *[file_record(path, role="stage_receipt") for path in runner.upstream_receipts],
            ],
        ),
        "outputs": outputs,
    }
    final = match_out / "run_manifests" / "broadcast_runner_event_inputs.json"
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return final


def write_point_ledger_manifest(
    *,
    video: Path,
    ledger: Path,
    ledger_report: Path,
    out_root: Path,
    match_id: str,
    fps: float,
    surface: str,
    stage_receipts: list[Path],
    person_model: Path,
    segmentation: str = "serve",
    segmentation_heads: tuple[Path, ...] = (),
) -> Path:
    provenance = build_provenance(
        root=REPO,
        mode=AUTOMATIC_MODE,
        source_videos=[file_record(video, role="source_broadcast")],
        models=[
            {
                "name": configured_vlm_model(),
                "role": "score_reader",
            },
            file_record(person_model, role="play_camera_person_detector"),
            # Which Stage-1 arm produced this ledger is otherwise unrecoverable from the
            # artifacts; `docs/wk1/s1_precision.md` had to reconstruct it from a shell history.
            *[file_record(head, role="stage_one_head") for head in segmentation_heads],
        ],
        configuration={
            "runner": "cv.pipeline.broadcast_runner",
            "stop_after": POINT_LEDGER_STOP,
            "match_id": match_id,
            "surface": surface,
            "source_fps": fps,
            "segmentation": segmentation,
            "vlm_url": os.environ.get("VLM_URL", "http://localhost:8399/v1/chat/completions"),
            "stages": [path.stem for path in stage_receipts],
        },
        reused_artifacts=[file_record(path, role="stage_receipt") for path in stage_receipts],
    )
    final = out_root / "run_manifests" / "broadcast_runner_point_ledger.json"
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_text(
        json.dumps(
            {
                "schema": "broadcast_point_ledger_run_v1",
                "provenance": provenance,
                "summary": ledger_point_summary(ledger),
                "outputs": [
                    file_record(ledger, role="automatic_point_ledger"),
                    file_record(ledger_report, role="automatic_point_ledger_report"),
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return final


def write_runner_parent_provenance(
    *,
    video: Path,
    out_root: Path,
    match_id: str,
    surface: str,
    person_model: Path,
) -> Path:
    """Write the automatic parent record required by pre-event gates.

    Point gates run before the final run manifest, so they cannot rely on a provenance
    record that is only written after reconstruction succeeds.
    """
    document = build_provenance(
        root=REPO,
        mode=AUTOMATIC_MODE,
        source_videos=[file_record(video, role="source_broadcast")],
        models=[file_record(person_model, role="play_camera_person_detector")],
        configuration={
            "runner": "cv.pipeline.broadcast_runner",
            "stage": "runner_parent",
            "match_id": match_id,
            "surface": surface,
            "labels_loaded": False,
        },
    )
    return write_provenance(out_root / "provenance.json", document)


INDEPENDENT_CONTACT_POSE_NAME = "player_pose_optional_contact_native_v1.csv"


def independent_contact_pose_requested(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "shared_s6_independent_contact_pose", False))


def reject_unusable_independent_contact_pose(args: argparse.Namespace) -> None:
    """Refuse the independent pose arm before any expensive stage runs.

    The artifact is only consumable by the shared S6 stage under an optional-contact policy,
    so a legacy backend or a policy that leaves optional contacts off would spend a full pose
    inference on evidence the run cannot admit.
    """
    if not independent_contact_pose_requested(args):
        return
    if getattr(args, "s6_backend", "legacy") != "shared_s6":
        raise ValueError("--shared-s6-independent-contact-pose requires --s6-backend shared_s6")
    policy = getattr(args, "shared_s6_policy", None)
    if policy is None:
        raise ValueError("--shared-s6-independent-contact-pose requires --shared-s6-policy")
    from cv.pipeline.s6_broadcast_backend import load_policy

    if load_policy(policy).get("optional_contacts", "off") != "on":
        raise ValueError(
            "--shared-s6-independent-contact-pose requires a policy with optional_contacts on"
        )


def write_independent_contact_pose_producer(
    *,
    receipt: Path,
    outputs: list[Path],
    parent_provenance: Path,
    upstream_receipts: list[Path],
    processing_video: Path,
    source_integrity: Path,
    pose_weights: Path,
    match_id: str,
    scope: str,
    coverage: Path,
    output: Path,
) -> Path:
    """Bind the pose this run actually produced to the stage receipt that produced it.

    The shared S6 validator resolves the named receipt, so the producer is assembled from the
    current receipt, its declared outputs and the run's own automatic parent -- never from a
    historical attestation. ``source_videos`` names only the picture stream the artifact was
    actually inferred over, so a pose produced from a healed normalized video cannot be admitted
    against the original broadcast. The parent source identity the run descends from, and the
    audit that produced the normalized stream, are bound alongside it.
    """
    parent = load_provenance(parent_provenance, require_automatic=True)
    processing_record = file_record(processing_video, role="processing_video")
    original_sources = [
        {**record, "role": "original_automatic_source_broadcast"}
        for record in parent["source_videos"]
        if record.get("sha256") != processing_record["sha256"]
    ]
    if original_sources:
        # Bind the audit that actually produced this stream, and only if it names it. An audit
        # of some other normalization would attest nothing about the pictures inferred here.
        audit = json.loads(source_integrity.read_text())
        if audit.get("cadence", {}).get("normalized_video") != processing_video.name:
            raise ValueError("source audit does not declare the processing video this run used")
    document = build_provenance(
        root=REPO,
        mode=AUTOMATIC_MODE,
        source_videos=[processing_record],
        models=[file_record(pose_weights, role="yolo26m_player_pose")],
        configuration={
            "runner": "cv.pipeline.broadcast_runner",
            "stage": "independent_contact_pose",
            "producer_module": "cv.pipeline.pose_player_crop",
            "producer_stage": "pose_player_crop",
            "receipt_written_by_runner_stage": True,
            "match_id": match_id,
            "observation_scope": scope,
            "artifact": outputs[0].name,
            "processing_video_is_parent_source": not original_sources,
            "actor_players_replaced": False,
            "labels_loaded": False,
            # The producer's own execution coverage, so a consumer reading this document
            # alone never mistakes the requested rows for covered evidence.
            "coverage": json.loads(coverage.read_text()),
        },
        reused_artifacts=[
            file_record(receipt, role="automatic_pose_inference_receipt"),
            *[file_record(path, role="automatic_pose_output") for path in outputs],
            file_record(coverage, role="automatic_pose_coverage"),
            file_record(parent_provenance, role="original_automatic_parent"),
            *original_sources,
            *[
                file_record(path, role="automatic_upstream_stage_receipt")
                for path in upstream_receipts
            ],
            *(
                [file_record(source_integrity, role="automatic_processing_source_audit")]
                if original_sources
                else []
            ),
        ],
    )
    return write_provenance(output, document)


def run_independent_contact_pose(
    *,
    runner: Runner,
    args: argparse.Namespace,
    match_out: Path,
    processing_video: Path,
    fps: float,
    pose_weights: Path,
    python: str,
) -> tuple[Path, Path]:
    """Produce and bind the optional-contact pose with the ordinary runner machinery.

    The artifact keeps its own name, so an enabled --physical-player-motion run and this arm
    write different files, hold different stage receipts and never overwrite one another.
    """
    boxes = match_out / f"player_boxes_{fps:g}_native_sided_v1.csv"
    pose = match_out / INDEPENDENT_CONTACT_POSE_NAME
    coordinates = pose.with_suffix(pose.suffix + ".coordinates.json")
    coverage = match_out / f"{pose.stem}.coverage.json"
    command = [
        python,
        "-m",
        "cv.pipeline.pose_player_crop",
        "--out",
        os.fspath(match_out),
        "--match-id",
        args.match_id,
        "--boxes",
        boxes.name,
        "--model",
        os.fspath(pose_weights),
        "--device",
        str(args.device),
        "--observation-scope",
        "retained_native",
        "--output",
        pose.name,
    ]
    frames = match_out / "audit_frames_native_1080"
    # Which pictures the retained scope may read is decided by the original per-clip
    # extraction receipts, so they are declared inputs: a re-extracted point invalidates
    # this stage even though the box stream and the frames directory name are unchanged.
    inventory = sorted(frames.glob("*/extraction_receipt.json"))
    dependencies = ("postseg_base",)
    runner.stage(
        "pose_player_crop",
        command,
        [
            pose_weights,
            boxes,
            boxes.with_suffix(boxes.suffix + ".coordinates.json"),
            match_out / "audit_frames_native_1080.coordinates.json",
            *inventory,
        ],
        [pose, coordinates, coverage],
        dependencies=dependencies,
    )
    producer = write_independent_contact_pose_producer(
        receipt=runner.receipts["pose_player_crop"],
        outputs=[pose, coordinates],
        parent_provenance=args.out / "provenance.json",
        # The ordinary frame, player and side receipts this scope descends from. They are
        # written by the canonical stage inside postseg_base, so an arm that ran without one
        # of them binds what it actually has rather than claiming a receipt it never wrote.
        upstream_receipts=[
            *[runner.receipts[name] for name in dependencies],
            *[
                path
                for name in ("frame_extraction", "player_detection", "player_side_association")
                if (path := receipt_path(match_out, name)).is_file()
            ],
        ],
        processing_video=processing_video,
        source_integrity=match_out / "source_integrity.json",
        pose_weights=pose_weights,
        match_id=args.match_id,
        scope="retained_native",
        coverage=coverage,
        output=match_out / f"{pose.stem}.producer.json",
    )
    # The shared-S6 opt-in returns before the final run manifest, so the arm records its own
    # configuration here rather than only inside the backend summary.
    manifest = match_out / "run_manifests" / "independent_contact_pose.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema": "independent_contact_pose_v1",
                "match_id": args.match_id,
                "observation_scope": "retained_native",
                "command": [str(value) for value in command],
                "stage_receipt": file_record(runner.receipts["pose_player_crop"]),
                "producer_provenance": file_record(producer),
                "outputs": [file_record(path) for path in (pose, coordinates)],
                "coverage": file_record(coverage),
                "extraction_receipts_declared": len(inventory),
                "forwarded_to": "cv.pipeline.s6_broadcast_backend.run_cached",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return pose, producer


def requested_evaluation_scope(args: argparse.Namespace) -> int | None:
    return evaluation_scope.validate_requested(getattr(args, "evaluation_first_attempts", None))


def reject_unusable_evaluation_scope(args: argparse.Namespace) -> None:
    """Refuse an unusable first-attempts scope before any expensive stage runs.

    The rejected combinations are the ones whose scope interactions this version does not
    establish: the physical branch aggregates body and camera dimension evidence across the
    whole match, and a second selection rule would make the executed cohort ambiguous.
    """
    requested = requested_evaluation_scope(args)
    if requested is None:
        return
    if getattr(args, "physical_player_motion", False):
        raise ValueError(
            "--evaluation-first-attempts does not support --physical-player-motion: "
            "camera_metric_refine and player_motion_physical aggregate dimension evidence "
            "across the match and their scope interactions are not established here"
        )
    maximum = getattr(args, "max_points", None)
    if maximum is not None and maximum != requested:
        raise ValueError(
            f"--max-points {maximum} contradicts --evaluation-first-attempts {requested}; "
            "omit it or pass the same value"
        )
    if getattr(args, "s6_backend", "legacy") != "shared_s6":
        raise ValueError(
            "--evaluation-first-attempts requires --s6-backend shared_s6: the legacy "
            "reconstruction path has no full-roster reporting contract for unselected rows"
        )


def run_shared_s6_backend(args: argparse.Namespace, processing_video: Path) -> Path:
    """Opt-in shared stage, with a separate report and original automatic inputs."""
    from cv.pipeline.s6_broadcast_backend import run_cached

    policy = getattr(args, "shared_s6_policy", None)
    if policy is None:
        raise ValueError("--s6-backend shared_s6 requires --shared-s6-policy")
    # Forwarded explicitly by the ordinary caller; the consumers also guard the root marker
    # on their own, so a standalone invocation against a scoped root cannot miss it.
    marker = evaluation_scope.scope_path(args.out)
    # Named only when a scope exists, so an ordinary full-broadcast dispatch is unchanged.
    scope_options = {"evaluation_scope": marker} if marker.is_file() else {}
    camera_options = {}
    if getattr(args, "shared_s6_camera_backend", "cached") == "window_metric":
        from cv.pipeline import window_camera_inference
        from cv.pipeline.camera_artifacts import RECONSTRUCTION_CAMERA_ARTIFACT

        stage_output = getattr(args, "shared_s6_output", None) or args.out / "shared_s6"
        camera_output = stage_output.with_name(stage_output.name + "_window_camera")
        producer = window_camera_inference.run(
            upstream_root=args.out,
            match_id=args.match_id,
            source_video=processing_video,
            output=camera_output,
            maximum=args.max_points,
            tape_measurement=getattr(args, "shared_s6_camera_tape_measurement", "hough_top"),
            **scope_options,
        )
        camera_options = dict(
            camera_projections=camera_output / RECONSTRUCTION_CAMERA_ARTIFACT,
            camera_provenance=producer,
        )
    pose_options = {}
    if independent_contact_pose_requested(args):
        # Only the flag admits the evidence: a stale namespace field can never activate it.
        pose = getattr(args, "shared_s6_optional_contact_pose", None)
        pose_provenance = getattr(args, "shared_s6_optional_contact_pose_provenance", None)
        if pose is None or pose_provenance is None:
            raise ValueError("independent contact pose was requested but this run produced none")
        pose_options = dict(
            optional_contact_pose=pose,
            optional_contact_pose_provenance=pose_provenance,
        )
    return run_cached(
        upstream_root=args.out,
        match_id=args.match_id,
        source_video=processing_video,
        policy_path=policy,
        output=getattr(args, "shared_s6_output", None) or args.out / "shared_s6",
        maximum=args.max_points,
        workers=getattr(args, "shared_s6_workers", 1),
        camera_name=(
            "camera_P_metric_v1.npz" if args.physical_player_motion else "camera_P_per_frame_v1.npz"
        ),
        execution_scope="broadcast_runner_automatic_upstream_and_shared_s6",
        **scope_options,
        **camera_options,
        **pose_options,
        event_emissions=getattr(args, "shared_s6_event_emissions", None),
        event_provenance=getattr(args, "shared_s6_event_provenance", None),
        observed_first_flight=getattr(args, "shared_s6_observed_first_flight", "off"),
        service_attempt_split=getattr(args, "shared_s6_service_attempt_split", "off"),
    )


def run(args: argparse.Namespace) -> Path:
    validate_event_marginal_threshold(getattr(args, "event_marginal_threshold", None))
    reject_unusable_independent_contact_pose(args)
    reject_unusable_evaluation_scope(args)
    evaluation_scope.preflight_root(args.out, args.match_id, requested_evaluation_scope(args))
    court_surface_witness = resolve_surface_witness_policy(
        getattr(args, "court_surface_witness", "off")
    )
    court_registration_mask = getattr(args, "court_registration_mask", "off")
    if court_registration_mask not in REGISTRATION_MASK_POLICIES:
        raise ValueError(f"unknown court registration mask {court_registration_mask!r}")
    interpolated_camera_policy = getattr(args, "interpolated_camera_policy", "off")
    if interpolated_camera_policy not in INTERPOLATED_CAMERA_POLICIES:
        raise ValueError(f"unknown interpolated camera policy {interpolated_camera_policy!r}")
    court_jobs = getattr(args, "court_jobs", None)
    resolved_court_jobs = resolve_court_jobs("postseg", court_jobs)
    lost_revival_body_history = bool(getattr(args, "lost_revival_body_history", False))
    overlap_fragment_recovery = bool(getattr(args, "overlap_fragment_recovery", False))
    metadata = probe(args.video)
    fps = float(metadata["fps"])
    match_out = args.out / args.match_id
    match_out.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    runner = Runner(match_out, args.resume)
    yolo_weights = data_root() / "models/pipeline/yolov8m.pt"
    pose_weights = data_root() / "models/pipeline/yolo26m-pose.pt"
    tracker_weights = tracker_root() / "pretrained_weights"
    wasb_weights = tracker_weights / "wasb_tennis_best.pth.tar"
    tracknet_weights = tracker_weights / "tracknetv2_tennis_best.pth.tar"
    write_runner_parent_provenance(
        video=args.video,
        out_root=args.out,
        match_id=args.match_id,
        surface=args.surface,
        person_model=yolo_weights,
    )

    runner.stage(
        "broadcast_source",
        [
            python,
            "-m",
            "cv.pipeline.broadcast_source",
            "--video",
            os.fspath(args.video),
            "--out",
            os.fspath(match_out),
            *(["--cadence-restore"] if getattr(args, "cadence_restore", False) else []),
        ],
        [args.video],
        [
            match_out / "source_integrity.json",
            match_out / "overlay_crops",
            match_out / "overlay_crops.coordinates.json",
        ],
    )
    source_integrity = json.loads((match_out / "source_integrity.json").read_text())
    processing_video, fps = processing_source(args.video, match_out, source_integrity)
    runner.stage(
        "play_camera",
        [
            python,
            "-m",
            "cv.pipeline.camera",
            "--video",
            os.fspath(args.video),
            "--out",
            os.fspath(match_out),
            "--fps",
            "1",
            "--source-frames",
            os.fspath(match_out / "overlay_crops"),
            "--person-model",
            os.fspath(yolo_weights),
            "--device",
            str(args.device),
        ],
        [yolo_weights],
        [match_out / "segments.csv", match_out / "labels.npz"],
        dependencies=("broadcast_source",),
    )
    runner.stage(
        "score_reading",
        [
            python,
            "-m",
            "cv.pipeline.score_vlm",
            "--out",
            os.fspath(match_out),
            "--fps",
            "1",
        ],
        [],
        [
            match_out / "score_runs.csv",
            match_out / "vlm_reads.jsonl",
            match_out / "score_vlm_manifest.json",
        ],
        dependencies=("broadcast_source",),
    )
    runner.stage(
        "serve_audio",
        [
            python,
            "-m",
            "cv.pipeline.serve_audio",
            "--video",
            os.fspath(args.video),
            "--out",
            os.fspath(match_out),
        ],
        [],
        [match_out / "serve_audio_peaks_v1.csv", match_out / "serve_audio_flux_v1.npz"],
        dependencies=("broadcast_source",),
    )
    ledger = match_out / "automatic_point_ledger_v1.csv"
    ledger_report = match_out / "automatic_point_ledger_v1.json"
    segmentation = args.segmentation
    view_head = serve_head = None
    if segmentation == "serve":
        view_head, serve_head = resolve_segmentation_models(args.segmentation_models)
    if segmentation == "serve":
        runner.stage(
            "shot_boundaries",
            [
                python,
                "-m",
                "cv.pipeline.shot_boundaries",
                # Frame indices become seconds at ``fps``, so decode the stream that has it.
                "--video",
                os.fspath(processing_video),
                "--out",
                os.fspath(match_out),
                "--fps",
                str(fps),
            ],
            [processing_video],
            [match_out / "shot_boundaries_v1.csv", match_out / "shot_signals_v1.npz"],
            dependencies=("broadcast_source",),
        )
        runner.stage(
            "native_view_frames",
            [
                python,
                "-m",
                "cv.pipeline.view_classifier",
                "frames",
                "--video",
                os.fspath(args.video),
                "--out",
                os.fspath(match_out),
                "--frame-fps",
                str(args.view_frame_fps),
            ],
            [args.video],
            [match_out / "frames_native.json"],
            dependencies=("broadcast_source",),
        )
        runner.stage(
            "frame_embeddings",
            [
                python,
                "-m",
                "cv.pipeline.view_classifier",
                "embed",
                "--out",
                os.fspath(match_out),
                "--device",
                f"cuda:{args.device}",
            ],
            [],
            [match_out / "frame_embeddings_v1.npz"],
            dependencies=("shot_boundaries", "native_view_frames"),
        )
        runner.stage(
            "shot_views",
            [
                python,
                "-m",
                "cv.pipeline.view_classifier",
                "predict",
                "--out",
                os.fspath(match_out),
                "--model",
                os.fspath(view_head),
            ],
            [view_head],
            [match_out / "shot_views_v1.csv"],
            dependencies=("frame_embeddings",),
        )
        runner.stage(
            "player_boxes",
            [
                python,
                "-m",
                "cv.pipeline.serve_detector",
                "boxes",
                "--out",
                os.fspath(match_out),
                "--weights",
                os.fspath(yolo_weights),
                "--device",
                str(args.device),
            ],
            [yolo_weights],
            [match_out / "player_boxes_v1.npz"],
            dependencies=("native_view_frames",),
        )
        runner.stage(
            "score_state",
            [
                python,
                "-m",
                "cv.pipeline.score_grammar",
                "--vlm-reads",
                os.fspath(match_out / "vlm_reads.jsonl"),
                "--best-of",
                str(args.best_of),
                "--tiebreak-target",
                str(args.tiebreak_target),
                "--deciding-tiebreak-target",
                str(args.deciding_tiebreak_target),
                *(["--points-only-boards"] if getattr(args, "points_only_boards", False) else []),
                "--out",
                os.fspath(match_out),
            ],
            [],
            [match_out / "score_state_v1.csv", match_out / "score_state_v1.json"],
            dependencies=("score_reading",),
        )
        runner.stage(
            "serve_attempts",
            [
                python,
                "-m",
                "cv.pipeline.serve_detector",
                "predict",
                "--out",
                os.fspath(match_out),
                "--model",
                os.fspath(serve_head),
                "--score-runs",
                os.fspath(match_out / "score_state_v1.csv"),
                *(
                    ["--observation-scope", "structural_context"]
                    if getattr(args, "attempt_observation_scope", "predicted")
                    == "structural_context"
                    else []
                ),
            ],
            [serve_head],
            [match_out / "serve_attempts_v1.csv"],
            dependencies=("shot_views", "player_boxes", "serve_audio", "score_state"),
        )
        ledger_command = [
            python,
            "-m",
            "cv.pipeline.point_ledger",
            "--mode",
            "serve",
            "--attempts",
            os.fspath(match_out / "serve_attempts_v1.csv"),
            "--score-state",
            os.fspath(match_out / "score_state_v1.csv"),
            "--serve-peaks",
            os.fspath(match_out / "serve_audio_peaks_v1.csv"),
            # One row per detected serve is the unit every downstream stage cuts a clip for;
            # the grouping columns say which of those rows are the same score point.
            "--rows",
            "attempts",
            "--point-continuation-gap",
            str(args.point_continuation_gap),
            "--output",
            os.fspath(ledger),
            "--report",
            os.fspath(ledger_report),
        ]
        ledger_dependencies = ("serve_attempts", "serve_audio")
    else:
        ledger_command = [
            python,
            "-m",
            "cv.pipeline.point_ledger",
            "--mode",
            "legacy",
            "--segments",
            os.fspath(match_out / "segments.csv"),
            "--score-runs",
            os.fspath(match_out / "score_runs.csv"),
            "--serve-peaks",
            os.fspath(match_out / "serve_audio_peaks_v1.csv"),
            "--output",
            os.fspath(ledger),
            "--report",
            os.fspath(ledger_report),
        ]
        ledger_dependencies = ("play_camera", "score_reading", "serve_audio")
    runner.stage(
        "point_ledger",
        ledger_command,
        [],
        [ledger, ledger_report],
        dependencies=ledger_dependencies,
    )
    if args.stop_after == POINT_LEDGER_STOP:
        return write_point_ledger_manifest(
            video=args.video,
            ledger=ledger,
            ledger_report=ledger_report,
            out_root=match_out,
            match_id=args.match_id,
            fps=fps,
            surface=args.surface,
            stage_receipts=runner.upstream_receipts,
            person_model=yolo_weights,
            segmentation=segmentation,
            segmentation_heads=((view_head, serve_head) if segmentation == "serve" else ()),
        )
    if args.event_model is None and args.stop_after != EVENT_INPUTS_STOP:
        raise ValueError("--event-model is required unless stopping before event inference")
    point_map = match_out / POINT_MAP_NAME
    manifest_path = args.out / "broadcast_manifest.json"
    requested_attempts = requested_evaluation_scope(args)
    scope = None
    if requested_attempts is not None:
        scope = evaluation_scope.build_scope(
            ledger=ledger,
            rows=evaluation_scope.ledger_rows(ledger),
            requested=requested_attempts,
            match_id=args.match_id,
            source_video=processing_video,
            upstream_root=args.out,
        )
        # Before anything is written: an existing root that means something else keeps
        # meaning it, and the run is sent to a new one.
        evaluation_scope.reject_incompatible_root(
            root=args.out,
            match_directory=match_out,
            requested=requested_attempts,
            selected_ids=list(scope["selected_point_ids"]),
            expected_scope=scope,
        )
        print(evaluation_scope.scope_summary_line(scope), flush=True)
    else:
        evaluation_scope.reject_incompatible_root(
            root=args.out, match_directory=match_out, requested=None, selected_ids=[]
        )
    # The per-point stages consume the selected work roster; the full broadcast manifest
    # stays behind for shared-S6 identity and full-ledger accounting.
    work_manifest_path = (
        manifest_path if scope is None else args.out / evaluation_scope.WORK_MANIFEST_NAME
    )

    def materialize() -> None:
        materialize_postseg(
            video=processing_video,
            match_out=match_out,
            ledger=ledger,
            manifest_path=manifest_path,
            match_id=args.match_id,
            fps=fps,
            surface=args.surface,
            scope=scope,
        )

    runner.stage(
        "postseg_materialization",
        [
            python,
            os.fspath(PIPELINE / "broadcast_runner.py"),
            "materialize",
            # The scope, the ledger it was cut from and the selected identities are part of
            # the stage identity, so a changed selection can never resume on this roster.
            *(evaluation_scope.scope_stage_identity(scope) if scope is not None else []),
        ],
        [],
        (
            [point_map, manifest_path]
            if scope is None
            else evaluation_scope.materialized_outputs(args.out, match_out)
        ),
        materialize,
        dependencies=("broadcast_source", "point_ledger"),
    )
    manifest = json.loads(work_manifest_path.read_text())
    match = manifest["matches"][0]
    runner.stage(
        "postseg_base",
        [
            python,
            os.fspath(PIPELINE / "canonical_runner.py"),
            "--manifest",
            os.fspath(work_manifest_path),
            "--out",
            os.fspath(args.out),
            "--device",
            str(args.device),
            *(["--court-jobs", str(court_jobs)] if court_jobs is not None else []),
            *(
                ["--court-surface-witness", court_surface_witness]
                if court_surface_witness != "off"
                else []
            ),
            *(
                ["--interpolated-camera-policy", interpolated_camera_policy]
                if interpolated_camera_policy != "off"
                else []
            ),
            *(["--lost-revival-body-history"] if lost_revival_body_history else []),
            *(["--overlap-fragment-recovery"] if overlap_fragment_recovery else []),
            *(
                ["--court-registration-mask", court_registration_mask]
                if court_registration_mask != "off"
                else []
            ),
        ],
        [yolo_weights],
        [
            match_out / "camera_P_per_frame_v1.npz",
            match_out / f"player_boxes_{fps:g}_native_sided_v1.csv",
        ],
        lambda: run_manifest(
            manifest,
            args.out,
            args.device,
            court_jobs=court_jobs,
            court_registration_mask=court_registration_mask,
            court_surface_witness=court_surface_witness,
            interpolated_camera_policy=interpolated_camera_policy,
            lost_revival_body_history=lost_revival_body_history,
            overlap_fragment_recovery=overlap_fragment_recovery,
        ),
        dependencies=("postseg_materialization",),
    )
    # Default-off ball-identity arms. Both appear in the declared stage command, so turning
    # either on changes the receipt fingerprint and the tracking stage is recomputed instead
    # of resumed from an output produced by the other arm.
    primary_ball_ownership = getattr(args, "primary_ball_ownership", False)
    anchor_extension = getattr(args, "anchor_extension", False)
    guide_sequence_association = getattr(args, "guide_sequence_association", False)
    runner.stage(
        "tracking",
        tracking_command(
            python,
            work_manifest_path,
            args.out,
            args.device,
            optimized_local_runtime=not getattr(args, "legacy_buffered_crop_refiner", False),
            primary_ball_ownership=primary_ball_ownership,
            anchor_extension=anchor_extension,
            guide_sequence_association=guide_sequence_association,
        ),
        [work_manifest_path, *tracking_model_inputs()],
        tracking_outputs(
            match_out,
            primary_ball_ownership=primary_ball_ownership,
            anchor_extension=anchor_extension,
        ),
        lambda: run_tracking_match(
            match,
            args.out,
            args.device,
            32,
            32,
            not getattr(args, "legacy_buffered_crop_refiner", False),
            True,
            primary_ball_ownership=primary_ball_ownership,
            anchor_extension=anchor_extension,
            guide_sequence_association=guide_sequence_association,
        ),
        dependencies=("postseg_base",),
    )
    active = args.out / "active_play_v1.json"
    cadence = args.out / "frame_cadence_audit_v1.json"
    court = args.out / "court_geometry_point_gate_v1.json"
    validity = args.out / "point_validity_gate_v1.json"
    camera_support = args.out / "reliable_per_frame_camera_support_v1.json"
    court_loss_policy = getattr(args, "active_play_court_loss_policy", "strict")
    native_actor_continuity = getattr(args, "active_play_native_actor_continuity", False)
    native_cut_continuity = getattr(args, "active_play_native_cut_continuity", False)
    court_anchor_camera = getattr(args, "active_play_court_anchor_camera", True)
    retained_play_scope = getattr(args, "retained_play_event_gate", False)
    gate_commands = point_gate_commands(
        python,
        args.out,
        work_manifest_path,
        court_loss_policy=court_loss_policy,
        native_actor_continuity=native_actor_continuity,
        native_cut_continuity=native_cut_continuity,
        retained_play_scope=retained_play_scope,
        court_anchor_camera=court_anchor_camera,
    )
    runner.stage(
        "point_gates",
        [
            python,
            *[
                os.fspath(PIPELINE / name)
                for name in (
                    "tracking_point_gate.py",
                    "active_play_gate.py",
                    "frame_cadence.py",
                    "court_geometry_point_gate.py",
                    "compose_point_validity_gate.py",
                    *(["camera_frame_support.py"] if retained_play_scope else []),
                )
            ],
            "--court-loss-policy",
            court_loss_policy,
            *(["--native-actor-continuity"] if native_actor_continuity else []),
            *(["--native-cut-continuity"] if native_cut_continuity else []),
            *(["--retained-play-scope"] if retained_play_scope else []),
            *(["--court-anchor-camera"] if court_anchor_camera else []),
        ],
        [
            DEFAULT_GATES,
            # Native qualification reads original pictures and their extraction clock.
            # Bind their live bytes so a cached gate cannot survive changed source evidence.
            *([match_out / "audit_frames_native_1080"] if native_cut_continuity else []),
            # The retained-play scope reads per-frame camera support from the automatic
            # frame transport, so bind its bytes rather than reading it on trust.
            *([match_out / "court_H_per_frame_v1.npz"] if retained_play_scope else []),
            # The court anchor picture is named by the court fit; bind that record.
            *([match_out / "court_topology_evidence_v1.json"] if court_anchor_camera else []),
        ],
        [
            active,
            args.out / "untouched_tracking_point_gate_v1.json",
            cadence,
            court,
            validity,
            *([camera_support] if retained_play_scope else []),
        ],
        lambda: _run_many(gate_commands),
        dependencies=("postseg_base", "tracking"),
    )
    pose_output = match_out / "player_pose_tracked_crop_native_v1.csv"
    physical_motion = match_out / "player_motion_physical_v1.jsonl"
    metric_camera = match_out / "camera_P_metric_v1.npz"
    body_dimensions = match_out / "player_body_dimensions_v1.json"
    if args.physical_player_motion:
        pose_commands = [
            [
                python,
                "-m",
                "cv.pipeline.pose_player_crop",
                "--out",
                os.fspath(match_out),
                "--match-id",
                args.match_id,
                "--active-play",
                os.fspath(active),
                "--boxes",
                f"player_boxes_{fps:g}_native_sided_v1.csv",
                "--model",
                os.fspath(pose_weights),
                "--device",
                str(args.device),
            ],
            [
                python,
                "-m",
                "cv.pipeline.camera_metric_refine",
                "--match-dir",
                os.fspath(match_out),
                "--match-id",
                args.match_id,
            ],
            [
                python,
                "-m",
                "cv.pipeline.player_motion_physical",
                "--match-dir",
                os.fspath(match_out),
                "--match-id",
                args.match_id,
                "--boxes-name",
                f"player_boxes_{fps:g}_native_sided_v1.csv",
                "--camera-name",
                metric_camera.name,
                "--dimension-locks",
                os.fspath(body_dimensions),
                "--cadence",
                os.fspath(cadence),
                "--body-prior-pose-strength",
                str(args.physical_body_prior_pose_strength),
                *(["--foot-contact-witness"] if args.physical_foot_contact_witness else []),
                *[
                    value
                    for path in args.physical_body_prior
                    for value in ("--body-prior", os.fspath(path))
                ],
            ],
        ]
        runner.stage(
            "physical_player_motion",
            [
                python,
                os.fspath(PIPELINE / "pose_player_crop.py"),
                os.fspath(PIPELINE / "camera_metric_refine.py"),
                os.fspath(PIPELINE / "player_motion_physical.py"),
            ],
            [
                pose_weights,
                active,
                cadence,
                match_out / f"player_boxes_{fps:g}_native_sided_v1.csv",
                PIPELINE / "player_biometrics.json",
                *args.physical_body_prior,
            ],
            [
                pose_output,
                pose_output.with_suffix(pose_output.suffix + ".coordinates.json"),
                metric_camera,
                body_dimensions,
                physical_motion,
            ],
            lambda: _run_many(pose_commands),
            dependencies=("point_gates", "postseg_base"),
        )
    if independent_contact_pose_requested(args):
        pose, producer = run_independent_contact_pose(
            runner=runner,
            args=args,
            match_out=match_out,
            processing_video=processing_video,
            fps=fps,
            pose_weights=pose_weights,
            python=python,
        )
        args.shared_s6_optional_contact_pose = pose
        args.shared_s6_optional_contact_pose_provenance = producer
    if args.stop_after == EVENT_INPUTS_STOP:
        return write_event_inputs_manifest(
            args=args,
            runner=runner,
            processing_video=processing_video,
            ledger=ledger,
            work_manifest=work_manifest_path,
        )
    event_crops = args.out / "event_crops_v1"
    emissions = args.out / "event_emissions.json"
    event_track = validate_event_track(getattr(args, "event_track", LEGACY_EVENT_TRACK))
    event_commands = [
        event_inference_command(
            python,
            args.out,
            args.event_model,
            device=args.device,
            supported_impulse_events=getattr(args, "supported_impulse_events", False),
            native_streak_fallback=getattr(args, "native_streak_fallback", False),
            model_accepted_tracking_held_events=getattr(
                args, "model_accepted_tracking_held_events", False
            ),
            native_net_evidence=getattr(args, "native_net_evidence", False),
            native_ground_evidence=getattr(args, "native_ground_evidence", False),
            event_court_geometry=getattr(args, "event_court_geometry", "point_static"),
            event_court_frame_missing=getattr(args, "event_court_frame_missing", "hold"),
            event_track=event_track,
            marginal_threshold=getattr(args, "event_marginal_threshold", None),
            live_shot_camera=getattr(args, "live_shot_camera", False),
        )
    ]
    runner.stage(
        "event_inference",
        event_commands[0],
        [
            args.event_model,
            work_manifest_path,
            active,
            validity,
            args.out / "untouched_tracking_point_gate_v1.json",
            *event_evidence_inputs(
                args.out,
                args.match_id,
                native_net_evidence=getattr(args, "native_net_evidence", False),
                native_ground_evidence=getattr(args, "native_ground_evidence", False),
                native_streak_fallback=getattr(args, "native_streak_fallback", False),
                event_track=event_track,
            ),
        ],
        [
            event_crops / "manifest.json",
            emissions,
            emissions.with_suffix(".manifest.json"),
            emissions.with_suffix(".predictions.npz"),
            *(
                [emissions.with_suffix(".streak_support.json")]
                if getattr(args, "native_streak_fallback", False)
                else []
            ),
            *(
                [emissions.with_suffix(".ground_evidence.json")]
                if getattr(args, "native_ground_evidence", False)
                else [emissions.with_suffix(".net_evidence.json")]
                if getattr(args, "native_net_evidence", False)
                else []
            ),
        ],
        lambda: _run_many(event_commands),
        dependencies=("point_gates", "tracking", "postseg_base"),
        # Only the optional picture-reading arm needs the extra check, so the default
        # stage keeps its exact previous reuse behaviour and argument list.
        **(
            {
                "cached_inputs": lambda: consumed_event_inputs_unchanged(
                    emissions.with_suffix(".manifest.json"),
                    emissions.with_suffix(".streak_support.json"),
                )
            }
            if getattr(args, "native_streak_fallback", False)
            else {}
        ),
    )
    contact_strikers = match_out / "contact_strikers_v1.json"
    runner.stage(
        "contact_strikers",
        [
            python,
            os.fspath(PIPELINE / "contact_striker.py"),
            "--match-dir",
            os.fspath(match_out),
            "--match-id",
            args.match_id,
            "--emissions",
            os.fspath(emissions),
            "--output",
            os.fspath(contact_strikers),
        ],
        [
            emissions,
            match_out / "court_H_per_point.npz",
            *sorted(match_out.glob("player_boxes_*_native_sided_v1.csv")),
            *([pose_output] if args.physical_player_motion else []),
        ],
        [contact_strikers],
        lambda: _run_many(
            [
                [
                    python,
                    "-m",
                    "cv.pipeline.contact_striker",
                    "--match-dir",
                    os.fspath(match_out),
                    "--match-id",
                    args.match_id,
                    "--emissions",
                    os.fspath(emissions),
                    "--output",
                    os.fspath(contact_strikers),
                ]
            ]
        ),
        dependencies=(
            "event_inference",
            "postseg_base",
            "tracking",
            *(("physical_player_motion",) if args.physical_player_motion else ()),
        ),
    )
    if args.stop_after == UPSTREAM_STOP:
        return match_out
    if getattr(args, "s6_backend", "legacy") == "shared_s6":
        return run_shared_s6_backend(args, processing_video)

    reconstruction = args.out / "reconstruction_3d.json"
    ledger_3d = args.out / "flight_ledger.json"
    reconstruction_points = reconstruction_point_arguments(ledger, args.match_id, args.max_points)
    selected_point_keys = set(reconstruction_points[1::2]) or None
    point_inputs_root = args.out / "reconstruction_point_inputs"
    point_input_layout = point_inputs_root / "manifest.json"
    point_reports_root = args.out / "reconstruction_point_reports"
    point_timings = args.out / "reconstruction_point_timings.json"
    reconstruction_camera_name = (
        metric_camera.name if args.physical_player_motion else "camera_P_per_frame_v1.npz"
    )
    reconstruction_pose_name = pose_output.name if args.physical_player_motion else None
    reconstruction_motion_name = physical_motion.name if args.physical_player_motion else None

    runner.stage(
        "reconstruction_inputs",
        [
            python,
            os.fspath(PIPELINE / "broadcast_runner.py"),
            "slice-reconstruction-inputs",
            os.fspath(point_inputs_root),
        ],
        [
            emissions,
            active,
            validity,
            args.out / "untouched_tracking_point_gate_v1.json",
            match_out / reconstruction_camera_name,
            match_out / "court_H_per_point.npz",
            match_out / "court_H_per_frame_v1.npz",
            match_out / "ball_track_joint_native1080_arc_augmented_v2.csv",
            *sorted(match_out.glob("player_boxes_*_native_sided_v1.csv")),
            *([pose_output, physical_motion] if args.physical_player_motion else []),
        ],
        [point_input_layout],
        lambda: prepare_reconstruction_point_inputs(
            out=args.out,
            manifest_path=work_manifest_path,
            emissions=emissions,
            destination=point_inputs_root,
            point_keys=selected_point_keys,
            pose_artifact_name=reconstruction_pose_name,
            physical_motion_name=reconstruction_motion_name,
            camera_artifact_name=reconstruction_camera_name,
        ),
        dependencies=(
            "event_inference",
            "postseg_base",
            "tracking",
            *(("physical_player_motion",) if args.physical_player_motion else ()),
        ),
    )
    ledger_command = [
        python,
        os.fspath(PIPELINE / "flight_ledger.py"),
        "--report",
        os.fspath(reconstruction),
        "--output",
        os.fspath(ledger_3d),
        "--csv",
        os.fspath(args.out / "flight_ledger.csv"),
    ]

    def reconstruct_points() -> None:
        run_point_reconstruction_layout(
            layout_path=point_input_layout,
            output=reconstruction,
            anchors_output_root=args.out / "reconstruction_anchors",
            reports_root=point_reports_root,
            timings_output=point_timings,
            workers=8,
        )
        subprocess.run(ledger_command, cwd=REPO, check=True)

    runner.stage(
        "reconstruction_3d",
        [
            python,
            os.fspath(PIPELINE / "reconstruct_3d.py"),
            os.fspath(PIPELINE / "flight_ledger.py"),
            "--point-local-input-layout",
            os.fspath(point_input_layout),
            *nightly_reconstruction_arguments(args.out),
        ],
        [point_input_layout],
        [reconstruction, ledger_3d, args.out / "flight_ledger.csv", point_timings],
        reconstruct_points,
        dependencies=("reconstruction_inputs",),
    )
    camera_manifest = json.loads(
        (match_out / "run_manifests" / "canonical_runner.json").read_text()
    )
    provenance = build_provenance(
        root=REPO,
        mode=AUTOMATIC_MODE,
        source_videos=[file_record(args.video, role="source_broadcast")],
        models=[
            {
                "name": configured_vlm_model(),
                "role": "score_reader",
            },
            file_record(yolo_weights, role="yolov8m_player_detector"),
            file_record(wasb_weights, role="wasb_ball_detector"),
            file_record(tracknet_weights, role="tracknetv2_ball_detector"),
            file_record(args.event_model, role="frozen_event_model"),
            {**bounce_reference.law_record(), "role": "court_bounce_law"},
            *(
                [file_record(pose_weights, role="yolo26m_player_pose")]
                if args.physical_player_motion
                else []
            ),
        ],
        configuration={
            "runner": "cv.pipeline.broadcast_runner",
            "match_id": args.match_id,
            "surface": args.surface,
            "source_fps": fps,
            "vlm_url": os.environ.get("VLM_URL", "http://localhost:8399/v1/chat/completions"),
            "court_jobs": resolved_court_jobs,
            "court_registration_mask": court_registration_mask,
            "court_surface_witness": court_surface_witness,
            "interpolated_camera_policy": interpolated_camera_policy,
            "lost_revival_body_history": lost_revival_body_history,
            "overlap_fragment_recovery": overlap_fragment_recovery,
            "stages": [path.stem for path in runner.upstream_receipts],
            "event_emission_mode": ("lossless_all_emissions_with_default_in_play_consumer_view"),
            "event_hypothesis_mode": "video_audio_decoder_best_path_with_abstention",
            "reconstruction_event_consumer_mode": "decoder_best_path_non_abstained",
            "attempt_observation_scope": getattr(args, "attempt_observation_scope", "predicted"),
            "shared_s6_service_attempt_split": getattr(
                args, "shared_s6_service_attempt_split", "off"
            ),
            **(
                {
                    "event_marginal_threshold": args.event_marginal_threshold,
                    "event_operating_point": "none",
                }
                if getattr(args, "event_marginal_threshold", None) is not None
                else {}
            ),
            "supported_impulse_events": getattr(args, "supported_impulse_events", False),
            # Declared only when selected, so a default run manifest is unchanged.
            **(
                {"native_streak_fallback": True}
                if getattr(args, "native_streak_fallback", False)
                else {}
            ),
            **(
                {"model_accepted_tracking_held_events": True}
                if getattr(args, "model_accepted_tracking_held_events", False)
                else {}
            ),
            "native_net_evidence": getattr(args, "native_net_evidence", False),
            "native_ground_evidence": getattr(args, "native_ground_evidence", False),
            "event_court_geometry": getattr(args, "event_court_geometry", "point_static"),
            "event_court_frame_missing": getattr(args, "event_court_frame_missing", "hold"),
            # Declared only when selected, so a default run manifest is unchanged.
            **(
                {"event_track": getattr(args, "event_track", LEGACY_EVENT_TRACK)}
                if getattr(args, "event_track", LEGACY_EVENT_TRACK) != LEGACY_EVENT_TRACK
                else {}
            ),
            "active_play_court_loss_policy": court_loss_policy,
            "active_play_native_actor_continuity": native_actor_continuity,
            # Declared only when selected, so a default run manifest is unchanged.
            **({"active_play_native_cut_continuity": True} if native_cut_continuity else {}),
            **({"active_play_court_anchor_camera": True} if court_anchor_camera else {}),
            **({"retained_play_event_gate": True} if retained_play_scope else {}),
            **({"live_shot_camera": True} if getattr(args, "live_shot_camera", False) else {}),
            "primary_ball_ownership": primary_ball_ownership,
            "ball_anchor_extension": anchor_extension,
            "guide_sequence_association": guide_sequence_association,
            "physical_player_motion": bool(args.physical_player_motion),
            "metric_camera_refinement": bool(args.physical_player_motion),
            "physical_body_priors": [os.fspath(path) for path in args.physical_body_prior],
            "physical_body_prior_pose_strength": args.physical_body_prior_pose_strength,
            "physical_foot_contact_witness": bool(args.physical_foot_contact_witness),
            "reconstruction_max_points": args.max_points,
        },
        reused_artifacts=[
            *[file_record(path, role="stage_receipt") for path in runner.upstream_receipts],
            *[
                file_record(path, role="automatic_player_body_prior")
                for path in args.physical_body_prior
            ],
        ],
        fallbacks=camera_manifest["provenance"]["fallbacks"],
    )
    final = match_out / "run_manifests" / "broadcast_runner.json"
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_text(
        json.dumps(
            {
                "schema": "broadcast_runner_v1",
                "provenance": provenance,
                "outputs": [
                    file_record(ledger, role="automatic_point_ledger"),
                    file_record(validity, role="point_validity_gate"),
                    file_record(emissions, role="typed_event_emissions"),
                    *(
                        [
                            file_record(pose_output, role="player_pose_2d"),
                            file_record(metric_camera, role="metric_player_height_camera"),
                            file_record(body_dimensions, role="locked_player_body_dimensions"),
                            file_record(physical_motion, role="physical_player_motion"),
                        ]
                        if args.physical_player_motion
                        else []
                    ),
                    file_record(reconstruction, role="reconstruction_3d"),
                    file_record(ledger_3d, role="flight_ledger"),
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--surface", choices=("hard", "clay", "grass"), required=True)
    parser.add_argument(
        "--event-model",
        type=Path,
        default=(
            processed_root() / "wk2_events" / "holdout_v2" / "event_video_model_all_development.pt"
        ),
    )
    parser.add_argument(
        "--event-marginal-threshold",
        type=float,
        help="Explicit event path-marginal threshold; bypasses the named checkpoint calibration. "
        "Bind external calibration evidence with the run. Default preserves the shipped policy.",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--active-play-court-loss-policy",
        choices=("strict", "preserve_supported_play"),
        default="strict",
        help="optional visual coverage preservation under static court-line support loss",
    )
    parser.add_argument(
        "--active-play-native-actor-continuity",
        action="store_true",
        help="optional image-space actor continuity for visual coverage",
    )
    parser.add_argument(
        "--active-play-native-cut-continuity",
        action="store_true",
        help="optional native qualification of sampled shot-cut proposals; default disabled",
    )
    # Default on, 2026-09-27: where the court fit's anchor picture falls on a sample the
    # longest-shot reference called a close-up, the anchor decides the play camera. 11 of
    # 149 opened clips change; labelled contacts never scored 41 -> 32 of 423; same-wave AA
    # and AA+cascade +3 flights (panel C 2, panel D 1), none lost, no extra accept.
    # Rollback: --no-active-play-court-anchor-camera, or flip this default.
    # Numbers: cv/experiments/active_span/FINAL_REPORT_active_span.md.
    parser.add_argument(
        "--active-play-court-anchor-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="the court fit's anchor picture decides the play camera where the "
        "longest-shot reference disagrees with it (default on)",
    )
    parser.add_argument(
        "--retained-play-event-gate",
        action="store_true",
        help="optional retained-play scope for points held only for "
        "phase:multiple_camera_shots: the point gate publishes the certified live-play "
        "spans and event acceptance is scoped to them; default disabled",
    )
    parser.add_argument(
        "--supported-impulse-events",
        action="store_true",
        help="optional two-wing event recovery; default remains disabled",
    )
    parser.add_argument(
        "--native-streak-fallback",
        action="store_true",
        help="optional measured native streak support for one wing of the two-wing "
        "event recovery; requires --supported-impulse-events, default disabled",
    )
    parser.add_argument(
        "--model-accepted-tracking-held-events",
        action="store_true",
        help="optional admission of model-accepted contacts/bounces whose only remaining "
        "veto is tracking-arc fit quality; the tracking hold is retained as evidence and "
        "the admitted row cannot certify a physical ending; default disabled",
    )
    parser.add_argument(
        "--native-net-evidence",
        action="store_true",
        help="optional native net decoder evidence; default disabled",
    )
    parser.add_argument(
        "--native-ground-evidence",
        action="store_true",
        help="optional native ground decoder evidence; default disabled",
    )
    parser.add_argument(
        "--live-shot-camera",
        action="store_true",
        help="run the event model on the wide-shot span of a clip held only for "
        "no_play_camera_shot; default disabled, same decision as the S6 release",
    )
    parser.add_argument(
        "--event-track",
        choices=(LEGACY_EVENT_TRACK, CURRENT_EVENT_TRACK),
        default=LEGACY_EVENT_TRACK,
        help="which composed ball track the event stage binds: the default legacy "
        "availability track, or the current S4 track read under its declared guide "
        "ancestry so only measured rows are observations",
    )
    parser.add_argument(
        "--event-court-geometry",
        choices=("point_static", "reliable_per_frame"),
        default="point_static",
        help="court transport used by the local event classifier",
    )
    parser.add_argument(
        "--event-court-frame-missing",
        choices=("hold", "point_static"),
        default="hold",
        help="explicit missing-frame policy for event court transport",
    )
    parser.add_argument(
        "--court-jobs",
        type=int,
        help="explicit CPU workers for court localization (default: 4)",
    )
    parser.add_argument(
        "--court-surface-witness",
        choices=SURFACE_WITNESS_POLICIES,
        default="off",
        help="illumination-aware court surface fallback after existing admission fails",
    )
    parser.add_argument(
        "--court-registration-mask",
        choices=REGISTRATION_MASK_POLICIES,
        default="off",
        help="source-feature masking for automatic per-frame court registration",
    )
    parser.add_argument(
        "--interpolated-camera-policy",
        choices=INTERPOLATED_CAMERA_POLICIES,
        default="off",
        help="native-picture qualification and same-anchor retry for interpolated cameras (default: off)",
    )
    parser.add_argument(
        "--lost-revival-body-history",
        action="store_true",
        help="default-off body-history guard at lost player-track revival; refuses an "
        "identity whose established measured body size class differs from the incoming "
        "detection, using the existing selection band and min_hits",
    )
    parser.add_argument(
        "--overlap-fragment-recovery",
        action="store_true",
        help="default-off recovery of measured player fragments from consistent duplicate-track overlap in "
        "player association; independent of the other association and camera options",
    )
    parser.add_argument(
        "--primary-ball-ownership",
        action="store_true",
        help="optional ball-identity ownership over the motion tracker's filter restarts; "
        "default disabled, and every tracked row is kept in the ownership side artifact",
    )
    parser.add_argument(
        "--anchor-extension",
        action="store_true",
        help="optional source-only extension of owned moving segment ends over the same "
        "candidate arms; requires --primary-ball-ownership, default disabled",
    )
    parser.add_argument(
        "--guide-sequence-association",
        action="store_true",
        help="optional two-pass motion-tracker association: re-solve short guide-opened "
        "detours that the same preceding primary immediately rejoins, over the full original "
        "measured candidate pool; default disabled",
    )
    parser.add_argument(
        "--legacy-buffered-crop-refiner",
        action="store_true",
        help="use the legacy unbounded decoded-frame crop refiner instead of streaming",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        help="run expensive 3D reconstruction only for the first N automatic ledger points",
    )
    parser.add_argument(
        "--evaluation-first-attempts",
        type=int,
        help="optional declared execution scope: after the complete automatic ledger is "
        "built, materialize and process only its first N rows in existing ledger order. "
        "The full ledger and the full broadcast manifest are unchanged; the remaining rows "
        "are reported as not_run_scope. Default absent, shared-S6 only, and a scoped root "
        "is a new measurement rather than a faster full-broadcast run",
    )
    parser.add_argument(
        "--attempt-observation-scope",
        choices=("predicted", "structural_context"),
        default="predicted",
        help="retain structural source context beyond tentative point ends",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--cadence-restore",
        action="store_true",
        help=(
            "restore isolated-pair repeat conversions (e.g. 25 fps doubled to 50 or pulled "
            "down to 30) to native cadence by the fitted conversion clock"
        ),
    )
    parser.add_argument(
        "--s6-backend",
        choices=("legacy", "shared_s6"),
        default="legacy",
        help="opt in to the shared local S6 stage on original automatic observations",
    )
    parser.add_argument(
        "--shared-s6-policy",
        type=Path,
        help="explicit composed S6 policy with a bound serve-location prior",
    )
    parser.add_argument(
        "--shared-s6-output",
        type=Path,
        help="new shared-S6 output directory (default: OUT/shared_s6)",
    )
    parser.add_argument("--shared-s6-workers", type=int, default=1)
    parser.add_argument("--shared-s6-observed-first-flight", choices=("off", "on"), default="off")
    parser.add_argument("--shared-s6-service-attempt-split", choices=("off", "on"), default="off")
    parser.add_argument(
        "--shared-s6-camera-backend", choices=("cached", "window_metric"), default="cached"
    )
    parser.add_argument(
        "--shared-s6-camera-tape-measurement",
        choices=("hough_top", "connected_pixels"),
        default="hough_top",
    )
    parser.add_argument("--shared-s6-event-emissions", type=Path)
    parser.add_argument("--shared-s6-event-provenance", type=Path)
    parser.add_argument(
        "--shared-s6-independent-contact-pose",
        action="store_true",
        help="run the ordinary pose producer over every retained native actor row and forward "
        "the result to the shared S6 stage as optional-contact evidence; default disabled, "
        "and independent of --physical-player-motion",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        choices=(3, 5),
        default=5,
        help="explicit match format for score decoding (legacy default: 5)",
    )
    parser.add_argument("--tiebreak-target", type=int, choices=(7, 10), default=7)
    parser.add_argument("--deciding-tiebreak-target", type=int, choices=(7, 10), default=10)
    parser.add_argument(
        "--points-only-boards",
        action="store_true",
        help="candidate points-only scoreboard decoding; not promoted for serve segmentation",
    )
    parser.add_argument(
        "--physical-player-motion",
        action="store_true",
        help="Run the candidate court-ray physical pose stage and expose racket branches to S6.",
    )
    parser.add_argument(
        "--physical-body-prior",
        type=Path,
        action="append",
        default=[],
        help=(
            "Explicit automatic player_body_prior_coco17_v1 JSONL from a temporal SMPL/MHR "
            "backend. Rejected if it contains human-derived inputs."
        ),
    )
    parser.add_argument(
        "--physical-body-prior-pose-strength",
        type=float,
        default=0.45,
        help="Soft articulation weight for the optional body prior; zero keeps racket FK only.",
    )
    parser.add_argument(
        "--physical-foot-contact-witness",
        action="store_true",
        help="Enable the experimental physical-pose grounded/airborne witness.",
    )
    parser.add_argument(
        "--segmentation",
        choices=("serve", "legacy"),
        default="serve",
        help=(
            "Stage-1 assembler. 'serve' anchors one attempt per detected serve and attaches "
            "the decoded score; it fails closed when its models are absent."
        ),
    )
    parser.add_argument(
        "--segmentation-models",
        type=Path,
        default=shipped_segmentation_models(),
        help=(
            "directory holding view_head_v1.json and serve_heads_v1.pkl; defaults to the "
            "shipped heads, and any other arm has to be named here"
        ),
    )
    parser.add_argument(
        "--point-continuation-gap",
        type=float,
        default=POINT_CONTINUATION_MAX_GAP_SECONDS,
        help=(
            "a serve this many seconds after the previous attempt ends opens a new score "
            "point; attempts closer than this on an unmoved score are one point served twice"
        ),
    )
    parser.add_argument(
        "--view-frame-fps",
        type=float,
        default=2.0,
        help="native full-resolution sampling cadence for the view and player features",
    )
    parser.add_argument(
        "--stop-after", choices=(POINT_LEDGER_STOP, EVENT_INPUTS_STOP, UPSTREAM_STOP)
    )
    args = parser.parse_args()
    try:
        validate_event_marginal_threshold(args.event_marginal_threshold)
    except ValueError as error:
        parser.error(str(error))
    if args.s6_backend == "shared_s6" and args.shared_s6_policy is None:
        parser.error("--s6-backend shared_s6 requires --shared-s6-policy")
    if args.shared_s6_independent_contact_pose and args.s6_backend != "shared_s6":
        parser.error("--shared-s6-independent-contact-pose requires --s6-backend shared_s6")
    if bool(args.shared_s6_event_emissions) != bool(args.shared_s6_event_provenance):
        parser.error("shared S6 event override requires both emissions and producer provenance")
    if args.shared_s6_workers < 1:
        parser.error("--shared-s6-workers must be positive")
    if args.court_jobs is not None and args.court_jobs < 1:
        parser.error("--court-jobs must be positive")
    if args.anchor_extension and not args.primary_ball_ownership:
        parser.error("--anchor-extension requires --primary-ball-ownership")
    if args.physical_body_prior and not args.physical_player_motion:
        parser.error("--physical-body-prior requires --physical-player-motion")
    if args.physical_foot_contact_witness and not args.physical_player_motion:
        parser.error("--physical-foot-contact-witness requires --physical-player-motion")
    if not 0.0 <= args.physical_body_prior_pose_strength <= 1.0:
        parser.error("--physical-body-prior-pose-strength must be between 0 and 1")
    if args.max_points is not None and args.max_points < 1:
        parser.error("--max-points must be at least one")
    # Validated at the command line as well as in `run`, so a mistyped scope costs nothing.
    if args.evaluation_first_attempts is not None:
        try:
            reject_unusable_evaluation_scope(args)
        except (ValueError, evaluation_scope.ScopeError) as error:
            parser.error(str(error))
    print(run(args))


if __name__ == "__main__":
    main()
