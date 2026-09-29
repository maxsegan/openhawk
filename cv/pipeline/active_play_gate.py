#!/usr/bin/env python3
"""Run the label-free active-play trimmer on a processed cohort.

Two purposes, both legitimate before the owner's audit returns because nothing here reads
labels:

  1. PRODUCTION READINESS. v2 is what production looks like -- native cadence (24 to 29.97
     fps), canonical-coordinate tracks over native frames, six unseen broadcasts. If the trimmer only works on the
     50 fps dev cohort it is not a stage.
  2. GATE CROSS-CHECK. The parallel agent froze a label-blind point gate (9 held / 39
     retained) from tracking-risk signals. This trimmer invalidates points from
     shot/phase signals. Overlap between two independent label-free verdicts is evidence
     both are seeing something real; disagreement is a review queue.

The command reads only processed tracking artifacts and never loads event labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cv.pipeline.active_play import (
    MINIMUM_LEAKAGE_TRIM_SECONDS,
    apply_leakage_trims,
    resolve_active_play,
)
from cv.pipeline.bounce_detect import GroundProjector, tape_line_ud
from cv.pipeline import resolution as res
from cv.pipeline.play_camera_leakage import (
    enrich_visual_support,
    load_match_inputs,
    propose_for_point,
)
from cv.pipeline.provenance import (
    AUTOMATIC_MODE,
    DIAGNOSTIC_MODE,
    ProvenanceError,
    build_provenance,
    file_record,
    load_provenance,
    validate_provenance,
    write_provenance,
)

TRACK_NAME = "ball_track_joint_native1080_arc_augmented_v2.csv"


def load_track_artifact(path: Path) -> dict[str, np.ndarray]:
    """Read authoritative coordinates into the phase model's fixed legacy-pixel units."""
    import csv
    from collections import defaultdict

    rows: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    manifest = res.read_coordinate_manifest(path)
    if manifest is None:
        raise ValueError(f"missing track coordinate manifest: {path}")
    with path.open() as handle:
        reader = csv.DictReader(handle)
        columns, source_size = res.coordinate_columns_and_size(
            reader.fieldnames or (),
            manifest,
            native_columns=("x_native", "y_native"),
            legacy_columns=("x", "y"),
        )
        for row in reader:
            digits = "".join(ch for ch in row.get("frame", "") if ch.isdigit())
            if digits:
                rows[row["clip"]].append(
                    (
                        float(int(digits)),
                        *res.scale_points(
                            [float(row[column]) for column in columns],
                            source_size,
                            res.LEGACY_TRACKING_SIZE,
                        ),
                    )
                )
    return {clip: np.array(sorted(values)) for clip, values in rows.items()}


def match_fps(match_dir: Path) -> float:
    meta = json.loads((match_dir / "audit_frames_native_1080.coordinates.json").read_text())
    return float(meta["fps"])


def net_rows_for_track(
    match_dir: Path,
    clip: str,
    track_path: Path,
    track: np.ndarray,
) -> np.ndarray | None:
    """Cord rows in the same phase-model units as load_track_artifact's output."""
    camera_path = match_dir / "camera_P_per_frame_v1.npz"
    if not camera_path.exists() or not len(track):
        return None
    try:
        manifest = res.read_coordinate_manifest(track_path)
        if manifest is None:
            return None
        image_size = res.FrameSize(**manifest["image_size"])
        projector = GroundProjector(
            str(camera_path),
            clip,
            image_size=image_size,
            artifact_size=res.LEGACY_TRACKING_SIZE,
        )
    except (ValueError, KeyError, OSError):
        return None
    rows = []
    for frame, image_x, _image_y in track:
        line = tape_line_ud(projector, int(round(frame)))
        finite = line[np.isfinite(line).all(axis=1)]
        if len(finite) < 2:
            return None
        order = np.argsort(finite[:, 0])
        rows.append(float(np.interp(image_x, finite[order, 0], finite[order, 1])))
    return np.asarray(rows, dtype=float)


def load_leakage_proposals(root: Path) -> tuple[dict[str, dict], list[Path]]:
    proposals = {}
    seen_matches = set()
    paths = sorted(root.glob("*__play_camera_leakage_v1.json"))
    for path in paths:
        report = json.loads(path.read_text())
        if report.get("schema") != "play_camera_leakage_v1":
            raise RuntimeError(f"unsupported leakage proposal schema in {path}")
        embedded_provenance = report.get("provenance")
        if embedded_provenance is not None:
            validate_provenance(embedded_provenance)
            if embedded_provenance["mode"] != AUTOMATIC_MODE:
                raise RuntimeError(f"non-automatic leakage proposal provenance in {path}")
        match_id = report["match_id"]
        if match_id in seen_matches:
            raise RuntimeError(f"duplicate leakage proposal match_id {match_id}")
        seen_matches.add(match_id)
        for clip, proposal in report.get("points", {}).items():
            key = f"{match_id}/{clip}"
            if key in proposals:
                raise RuntimeError(f"duplicate leakage proposal point {key}")
            proposals[key] = {**proposal, "schema": report["schema"]}
    return proposals, paths


def _court_anchor(match_dir: Path, clip_dir: Path) -> Path | None:
    from cv.pipeline.court_anchor_camera import anchor_frame_name

    name = anchor_frame_name(match_dir, clip_dir.name)
    return None if name is None else clip_dir / name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-inputs",
        action="store_true",
        help="accept explicit diagnostic parent ancestry and retain it in this output; "
        "never eligible as automatic whole-video inference",
    )
    parser.add_argument("--tracking-gate", type=Path)
    parser.add_argument(
        "--leakage-root",
        type=Path,
        help=(
            "explicit automatic play_camera_leakage_v1 proposal directory; when omitted, "
            "proposals are computed from the regenerated active spans"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--court-loss-policy",
        choices=("strict", "preserve_supported_play"),
        default="strict",
        help="optional preservation of independently supported visual play under static court loss",
    )
    parser.add_argument(
        "--phase-profile",
        choices=("baseline", "net_guarded_v1"),
        default="baseline",
        help="explicit phase calibration profile; baseline remains canonical",
    )
    parser.add_argument(
        "--native-actor-continuity",
        action="store_true",
        help="optional image-space actor witness across short held-camera gaps",
    )
    parser.add_argument(
        "--native-cut-continuity",
        action="store_true",
        help="optional native qualification of sampled cut proposals; default disabled",
    )
    parser.add_argument(
        "--court-anchor-camera",
        action="store_true",
        help="optional: let the court fit's anchor picture decide the play camera where the "
        "longest-shot reference disagrees with it; default disabled",
    )
    args = parser.parse_args()
    if args.native_actor_continuity and args.court_loss_policy != "preserve_supported_play":
        parser.error("--native-actor-continuity requires preserve_supported_play")

    gate_path = args.tracking_gate
    if gate_path is not None and gate_path.exists():
        gate = json.loads(gate_path.read_text())
        held = {
            (entry["match_id"], entry["clip"])
            for entry in gate["rows"]
            if entry.get("decision") == "hold"
        }
    else:
        held = set()
    if args.leakage_root:
        frozen_proposals, leakage_paths = load_leakage_proposals(args.leakage_root)
    else:
        frozen_proposals, leakage_paths = None, []
    parent_provenance_path = args.processed / "provenance.json"
    parent_provenance = load_provenance(
        parent_provenance_path, require_automatic=not args.diagnostic_inputs
    )
    if args.diagnostic_inputs and parent_provenance["mode"] != DIAGNOSTIC_MODE:
        raise ProvenanceError("--diagnostic-inputs requires an explicitly diagnostic parent")
    inherited_assistance = (
        {
            field: parent_provenance[field]
            for field in ("human_inputs", "reviewed_inputs", "manual_overrides")
        }
        if args.diagnostic_inputs
        else {}
    )
    reused_artifacts = [file_record(parent_provenance_path, role="parent_provenance")]
    if gate_path is not None and gate_path.is_file():
        reused_artifacts.append(file_record(gate_path, role="tracking_gate"))
    reused_artifacts.extend(file_record(path, role="leakage_proposals") for path in leakage_paths)
    provenance = build_provenance(
        root=Path(__file__).resolve().parents[2],
        mode=DIAGNOSTIC_MODE if args.diagnostic_inputs else AUTOMATIC_MODE,
        **inherited_assistance,
        source_videos=parent_provenance["source_videos"],
        models=parent_provenance["models"],
        configuration={
            "stage": "active_play_gate",
            "processed": args.processed,
            "tracking_gate": gate_path if gate_path is not None and gate_path.is_file() else None,
            "phase_profile": args.phase_profile,
            "court_loss_policy": args.court_loss_policy,
            "native_actor_continuity": args.native_actor_continuity,
            "native_cut_continuity": args.native_cut_continuity,
            # Declared only when selected, so a default provenance keeps its keys.
            **({"court_anchor_camera": True} if args.court_anchor_camera else {}),
            "leakage_proposal_mode": "reused" if args.leakage_root else "computed",
            "leakage_root": args.leakage_root,
            "minimum_leakage_trim_seconds": MINIMUM_LEAKAGE_TRIM_SECONDS,
            "labels_loaded": False,
        },
        reused_artifacts=reused_artifacts,
        fallbacks=parent_provenance["fallbacks"],
    )

    results = {}
    agree_hold = my_hold = their_hold = 0
    for match_dir in sorted(args.processed.iterdir()):
        track_path = match_dir / TRACK_NAME
        if not match_dir.is_dir() or not track_path.exists():
            continue
        fps = match_fps(match_dir)
        tracks = load_track_artifact(track_path)
        leakage_inputs = (
            load_match_inputs(match_dir, native_actor_continuity=args.native_actor_continuity)
            if frozen_proposals is None or args.court_loss_policy == "preserve_supported_play"
            else None
        )
        for clip_dir in sorted((match_dir / "audit_frames_native_1080").glob("pt*")):
            paths = sorted(clip_dir.glob("f_*.jpg"))
            track = tracks.get(clip_dir.name, np.empty((0, 3)))
            net_rows = (
                net_rows_for_track(
                    match_dir,
                    clip_dir.name,
                    track_path,
                    track,
                )
                if args.phase_profile == "net_guarded_v1"
                else None
            )
            decision = resolve_active_play(
                paths,
                track,
                fps=fps,
                net_row=net_rows,
                phase_profile=args.phase_profile,
                native_cut_continuity=args.native_cut_continuity,
                **(
                    {"court_anchor": _court_anchor(match_dir, clip_dir)}
                    if args.court_anchor_camera
                    else {}
                ),
            )
            key = f"{match_dir.name}/{clip_dir.name}"
            gate_held = (match_dir.name, clip_dir.name) in held
            entry = decision.as_dict() | {"gate_held": gate_held}
            if frozen_proposals is None:
                _, leakage_proposal = propose_for_point(
                    match_dir,
                    clip_dir.name,
                    entry,
                    inputs=leakage_inputs,
                )
                leakage_proposal["schema"] = "play_camera_leakage_v1"
            else:
                if key not in frozen_proposals:
                    raise RuntimeError(f"missing leakage proposal for {key}")
                leakage_proposal = frozen_proposals[key]
                if (
                    args.court_loss_policy == "preserve_supported_play"
                    and not args.native_actor_continuity
                ):
                    leakage_proposal = enrich_visual_support(
                        match_dir,
                        clip_dir.name,
                        leakage_proposal,
                        inputs=leakage_inputs,
                        entry=entry,
                    )
            if args.native_actor_continuity:
                leakage_proposal = enrich_visual_support(
                    match_dir, clip_dir.name, leakage_proposal, inputs=leakage_inputs, entry=entry
                )
            entry = apply_leakage_trims(
                entry, leakage_proposal, court_loss_policy=args.court_loss_policy
            )
            results[key] = entry
            my_hold += not entry["point_valid"]
            their_hold += gate_held
            agree_hold += not entry["point_valid"] and gate_held
            trim_note = (
                f"[{decision.trim[0]:.0f},{decision.trim[1]:.0f}] cut {decision.trimmed_fraction:.0%}"
                if decision.trim
                else "none"
            )
            print(
                f"{key:<46} fps={fps:<6} valid={str(entry['point_valid']):<6} "
                f"trim={trim_note:<22} gate_held={gate_held} "
                f"leakage={entry['leakage_trim']['decision']} {';'.join(entry['reasons'])}"
            )
        if leakage_inputs is not None:
            leakage_inputs.assert_unchanged()
            if args.native_actor_continuity:
                provenance["reused_artifacts"].extend(
                    file_record(path, role="native_actor_continuity_input")
                    for path in leakage_inputs.actor_inputs.paths
                )

    n = len(results)
    print(
        f"\npoints {n} | my invalid {my_hold} ({my_hold / max(n, 1):.0%}) | "
        f"gate held {their_hold} | both-hold overlap {agree_hold}"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    provenance_path = args.out.with_name(f"{args.out.stem}.provenance.json")
    write_provenance(provenance_path, provenance)
    print(f"wrote {args.out}")
    print(f"wrote {provenance_path}")


if __name__ == "__main__":
    main()
