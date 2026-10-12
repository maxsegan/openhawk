"""Run automatic S2 on original ledger windows and native extraction frames.

This is the local window-camera producer used by broadcast_runner's optional
shared-S6 camera backend. It needs no labels, fitted trajectories or model calls.
Every source ledger row remains in the report, including unavailable windows.

python -m cv.pipeline.window_camera_inference --upstream-root UP --match-id MATCH
  --source-video VIDEO --max-points 8 --output NEW
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

from cv.pipeline import artifact_cache, camera_artifacts, paths, provenance
from cv.pipeline import evaluation_scope as scope_module
from cv.pipeline.s6_automatic_observations import _native_inventory, validate_ancestry
from cv.pipeline.s6_broadcast_backend import discover_upstream, source_pts_cache

REPO = Path(__file__).resolve().parents[2]


def _read(path):
    return json.loads(path.read_text())


def _save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_windows(
    manifest: dict,
    ledger: list[dict],
    match_id: str,
    maximum: int | None,
    scope: dict | None = None,
):
    """Original source order only; no fitted outcome or per-window choices.

    A declared execution scope limits how many original rows this producer attempts; every
    original row stays in the report, out-of-scope rows as ``not_run_scope``.
    """
    matches = [row for row in manifest["matches"] if row["id"] == match_id]
    if len(matches) != 1 or (maximum is not None and maximum < 1):
        raise ValueError("one original match and positive optional cap required")
    ids = [int(row["pt"]) for row in ledger]
    if len(set(ids)) != len(ids) or set(ids) != set(matches[0]["point_ids"]):
        raise ValueError("original ledger and manifest roster differ")
    limit = maximum if scope is None else scope_module.selected_limit(scope, maximum)
    unselected = "not_run_cap" if scope is None else "not_run_scope"
    rows = [
        dict(
            clip=f"pt{int(row['pt']):04d}",
            ledger_row=row,
            status="pending" if limit is None or i < limit else unselected,
        )
        for i, row in enumerate(ledger)
    ]
    return matches[0], rows


def combine_windows(files: list[Path], output: Path) -> None:
    """Concatenate actual native camera rows; never fill an absent window."""
    if not files:
        np.savez_compressed(
            output,
            clips=np.asarray([], dtype=str),
            frames=np.asarray([], dtype=int),
            P=np.empty((0, 3, 4)),
            reliable=np.asarray([], dtype=bool),
            source=np.asarray([], dtype=str),
            confidence=np.asarray([], dtype=float),
        )
        return
    arrays = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            arrays.append({k: data[k] for k in data.files})
    if any(row.keys() != arrays[0].keys() for row in arrays):
        raise ValueError("window camera schema differs")
    joined = {k: np.concatenate([row[k] for row in arrays]) for k in arrays[0]}
    keys = list(zip(joined["clips"].tolist(), joined["frames"].tolist(), strict=True))
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate original camera frame")
    np.savez_compressed(output, **joined)


def run(
    *,
    upstream_root: Path,
    match_id: str,
    source_video: Path,
    output: Path,
    maximum: int | None = None,
    tape_measurement: str = "hough_top",
    evaluation_scope: Path | None = None,
) -> Path:
    if Path(match_id).name != match_id or not match_id:
        raise ValueError("safe match identity required")
    if tape_measurement not in {"hough_top", "connected_pixels"}:
        raise ValueError("unknown tape measurement")
    if output.exists():
        raise FileExistsError("new camera producer output required")
    paths.data_relative(output)
    parent = upstream_root / match_id
    manifest = upstream_root / "broadcast_manifest.json"
    ledger = parent / "automatic_point_ledger_v1.csv"
    with ledger.open(newline="") as stream:
        ledger_rows = list(csv.DictReader(stream))
    # The root marker is authoritative, so a standalone call against a scoped upstream root
    # is guarded even when no scope is forwarded.
    scope, _ = scope_module.guard_consumer_root(
        upstream_root=upstream_root,
        match_id=match_id,
        ledger=ledger,
        rows=ledger_rows,
        maximum=maximum,
        source_video=source_video,
    )
    if evaluation_scope is not None and (
        scope is None or Path(evaluation_scope) != scope_module.scope_path(upstream_root)
    ):
        raise ValueError("forwarded execution scope is not this upstream root's scope")
    match, rows = source_windows(_read(manifest), ledger_rows, match_id, maximum, scope)
    receipts, parents = discover_upstream(upstream_root, match_id)
    output.mkdir(parents=True)
    proof = provenance.write_provenance(
        output / "input_provenance.json",
        provenance.build_provenance(
            root=REPO,
            mode=provenance.AUTOMATIC_MODE,
            source_videos=[provenance.file_record(source_video)],
            configuration=dict(
                entrypoint="cv.pipeline.window_camera_inference", stage="input_admission"
            ),
            reused_artifacts=[
                *[provenance.file_record(p, role="automatic_parent") for p in parents],
                *[provenance.file_record(p, role="automatic_stage_receipt") for p in receipts],
            ],
        ),
    )
    validate_ancestry(
        SimpleNamespace(source_video=source_video, ancestry=(proof,)),
        [
            manifest,
            ledger,
            *([scope_module.scope_path(upstream_root)] if scope is not None else []),
        ],
    )
    fps = float(match["source_fps"])
    pts, pts_receipt = source_pts_cache(source_video, fps, upstream_root / "source_pts_cache")
    inputs = [manifest, ledger, proof, pts, pts_receipt]
    if scope is not None:
        # Declared as a consumed input, so a producer built under one scope is never
        # mistaken for one built under another.
        inputs.append(scope_module.scope_path(upstream_root))
    for row in rows:
        if row["status"] != "pending":
            continue
        frames = parent / "audit_frames_native_1080" / row["clip"]
        extraction = frames / "extraction_receipt.json"
        row["frames_directory"] = provenance.portable_path(frames)
        if extraction.is_file():
            inputs.append(extraction)
            row["extraction_receipt"] = provenance.file_record(extraction)
        try:
            images = _native_inventory(
                SimpleNamespace(
                    clip=row["clip"],
                    extraction_receipt=extraction,
                    frames_directory=frames,
                    source_video=source_video,
                    source_pts=pts,
                ),
                fps,
                row["ledger_row"],
            )
            row["native_images"] = images
            inputs.extend(frames / f"f_{image['frame']:04d}.jpg" for image in images)
        except (ValueError, OSError, KeyError) as error:
            row.update(status="preparation_failed", reason=f"{type(error).__name__}: {error}")
    inventory = output / "native_inventory.json"
    _save(inventory, dict(roster_attempts=len(rows), rows=rows))
    inputs.append(inventory)
    configuration = dict(
        entrypoint="cv.pipeline.window_camera_inference",
        receipt_timing="construction_time",
        match_id=match_id,
        maximum=maximum,
        roster_attempts=len(rows),
        selection="original automatic ledger source order; complete original native extraction window",
        height_source=camera_artifacts.DEFAULT_WINDOW_HEIGHT_SOURCE,
        tape_measurement=tape_measurement,
        registration_stride=camera_artifacts.DEFAULT_WINDOW_REGISTRATION_STRIDE,
        model_calls=0,
        player_height_observations="not supplied",
        all_roster_rows_retained=True,
        # Declared only when a scope is bound, so a default producer's identity is unchanged.
        **({"processing_scope": scope_module.describe(scope)} if scope is not None else {}),
    )
    command = [sys.executable, "-m", "cv.pipeline.window_camera_inference"]
    identity = artifact_cache.stage_identity(
        stage="window_camera_inference",
        command=command,
        inputs=inputs,
        upstream_receipts=receipts,
        configuration=configuration,
    )
    _save(output / "CONSTRUCTION.json", identity)
    produced = []
    files = []
    started = time.monotonic()
    for row in rows:
        if row["status"] != "pending":
            continue
        frames = parent / "audit_frames_native_1080" / row["clip"]
        frame_ids = [image["frame"] for image in row["native_images"]]
        frame_paths = [frames / f"f_{frame:04d}.jpg" for frame in frame_ids]
        try:
            camera = camera_artifacts.solve_window_camera(
                frame_paths,
                frame_ids,
                **(
                    {}
                    if tape_measurement == "hough_top"
                    else {"tape_measurement": tape_measurement}
                ),
            )
            if isinstance(camera, dict):
                row.update(status="held", camera_refusal=camera)
                continue
            destination = output / "windows" / row["clip"]
            document = camera_artifacts.write_window_camera(
                destination,
                match_id,
                row["clip"],
                camera,
                inputs=[row["extraction_receipt"], provenance.file_record(inventory)],
                configuration=configuration,
            )
            row.update(
                status="completed",
                supported_frames=document["supported"],
                total_frames=document["total"],
                camera=provenance.file_record(destination / "cameras.json"),
            )
            files.append(destination / camera_artifacts.RECONSTRUCTION_CAMERA_ARTIFACT)
            produced.extend(
                [
                    destination / "cameras.json",
                    destination / camera_artifacts.RECONSTRUCTION_CAMERA_ARTIFACT,
                    destination / camera_artifacts.FRAME_TRACK_ARTIFACT,
                ]
            )
        except (ValueError, OSError, KeyError, np.linalg.LinAlgError) as error:
            row.update(status="execution_failed", reason=f"{type(error).__name__}: {error}")
    camera_path = output / camera_artifacts.RECONSTRUCTION_CAMERA_ARTIFACT
    combine_windows(files, camera_path)
    report = output / "REPORT.json"
    _save(
        report,
        dict(
            configuration=configuration,
            rows=rows,
            elapsed_seconds=time.monotonic() - started,
            gate_or_useful_yield=None,
            scope="automatic S2 output only; S6 not run by this producer",
        ),
    )
    produced.extend([camera_path, report])
    receipt = artifact_cache.write_stage_receipt(
        out_dir=output,
        stage="window_camera_inference",
        command=command,
        inputs=inputs,
        outputs=produced,
        upstream_receipts=receipts,
        configuration=configuration,
        expected_identity=identity,
    )
    producer = provenance.write_provenance(
        output / "provenance.json",
        provenance.build_provenance(
            root=REPO,
            mode=provenance.AUTOMATIC_MODE,
            source_videos=[provenance.file_record(source_video)],
            configuration=configuration,
            reused_artifacts=[
                provenance.file_record(proof, role="automatic_input_provenance"),
                provenance.file_record(receipt, role="automatic_camera_inference_receipt"),
                *[provenance.file_record(p, role="automatic_camera_output") for p in produced],
            ],
        ),
    )
    validate_ancestry(SimpleNamespace(source_video=source_video, ancestry=(producer,)), produced)
    return producer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream-root", "source-video", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--max-points", dest="maximum", type=int)
    parser.add_argument(
        "--tape-measurement", choices=("hough_top", "connected_pixels"), default="hough_top"
    )
    print(run(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
