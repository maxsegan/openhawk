"""Measure defects in frozen human-conditioned prefixes, never certify 3D accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cv.experiments.connected_shooting import human_replay, model, physical_audit
from cv.pipeline.provenance import file_record, git_record


def source_identity_matches(source: dict) -> bool:
    """Verify the explicitly bound file, independent of the auditing checkout's root."""
    current = file_record(Path(source["resolved_path"]))
    # Portable path names are contextual: a sibling worktree's repository file
    # becomes external when audited from here. The resolved binding is unchanged.
    return all(current[key] == source["record"][key] for key in ("sha256", "bytes"))


def measure(record: dict, scene: model.Scene, heldout: model.Scene, bounces: list) -> dict:
    frames = tuple(
        np.unique(np.r_[np.arange(a, b, scene.fps / 240), b])
        for a, b in zip(scene.contact_frames, scene.contact_frames[1:])
    )
    replay = model.chain(scene, record["parameters"], query_frames=frames)
    if len(replay) != len(record["dense_flights"]) or any(
        not np.array_equal(fit["positions"], saved["positions"])
        for fit, saved in zip(replay, record["dense_flights"], strict=True)
    ):
        raise ValueError("current replay differs from frozen human-prefix path")
    flights = []
    for i, (fit, fs, bounce) in enumerate(zip(replay, frames, bounces, strict=True)):
        flights.append(
            {
                "flight_index": i,
                "modeled_bounces": len(fit["bounces"]),
                "human_bounces": 1,
                "bounce_frame_errors": [b["frame"] - bounce for b in fit["bounces"]],
                "net_crossings": physical_audit.crossings(fs, fit["positions"]),
                "minimum_height_m": float(np.min(fit["positions"][:, 2])),
                "maximum_speed_mps": float(np.max(np.linalg.norm(fit["velocities"], axis=1))),
                "impact_passivity": [
                    physical_audit.impact_passivity(b, fit["end_frame"], scene.fps)
                    for b in fit["bounces"]
                ],
            }
        )
    observations = human_replay.native_observations(scene, heldout, record["parameters"])
    events = []
    for event in record["evidence"]["human_events"]:
        # The old labeler could prefill even a "new" event from a track. Its
        # CSV does not distinguish this from a manual click. This is coordinate
        # disagreement only, never independent point-label accuracy.
        frame = float(event["labeled_frame"])
        if frame not in observations or not all(
            event.get(k) for k in ("labeled_x540", "labeled_y540")
        ):
            events.append({"frame": frame, "status": "missing_native_observation_or_click"})
            continue
        owner = np.array([float(event["labeled_x540"]), float(event["labeled_y540"])]) * 2
        automatic, prediction = observations[frame]
        events.append(
            {
                "frame": frame,
                "kind": event["event_type"],
                "status": "measured_archived_coordinate_disagreement",
                "track_to_archived_coordinate_px": float(np.linalg.norm(automatic - owner)),
                "fit_to_archived_coordinate_px": float(np.linalg.norm(prediction - owner)),
                "coordinate_origin": "not_preserved_by_csv",
                "independent_coordinate_click_verified": False,
            }
        )
    return {
        "flights": flights,
        "events": events,
        "maximum_contact_join_m": max(
            (
                float(np.linalg.norm(a["end_xyz"] - b["start_xyz"]))
                for a, b in zip(replay, replay[1:])
            ),
            default=0.0,
        ),
        "bounce_count_mismatch_flights": sum(f["modeled_bounces"] != 1 for f in flights),
        "net_penetrations": sum(c["penetration"] for f in flights for c in f["net_crossings"]),
        "complete_point_accepted": False,
        "independent_3d_accuracy_measured": False,
    }


def main() -> None:
    from dataclasses import replace

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--render-dir", type=Path, help="new directory for frozen-fit native overlays; no refitting"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.report.read_text())
    if report["schema"] != "connected_human_prefix_replay_v1" or report["status"] != "complete":
        raise ValueError("completed human replay required")
    bindings = report["inputs"]
    for source in bindings:
        if not source_identity_matches(source):
            raise ValueError("human replay source identity changed")
    config = human_replay.effective_config(
        json.loads(Path(report["configuration"]["cases"]).read_text()),
        report["configuration"].get("candidate_top1"),
        report["configuration"].get("camera_input"),
    )
    if config["native_size"] != [1920, 1080]:
        raise ValueError("archived x540 click audit requires explicit native 1920x1080 inputs")
    root = Path(report["configuration"]["match_root"])
    events = Path(report["configuration"]["events"])
    cases = {config["match_id"] + "__" + c["clip"]: c for c in config["cases"]}
    if args.render_dir:
        args.render_dir.mkdir(parents=True, exist_ok=False)
        for case in cases.values():
            inventory = human_replay.frame_inventory(
                root / config["inputs"]["frames"] / case["clip"]
            )
            if inventory != report["frame_inventories"][case["clip"]]:
                raise ValueError("render source exposures differ from frozen replay")
    case_paths = []
    for row in report["results"]:
        case_paths.append(
            args.report.parent / f"{cases[row['case']]['clip']}_{row['rebound_mode']}.json"
        )
    records = [
        args.report,
        Path(__file__),
        Path(physical_audit.__file__),
        Path(human_replay.__file__),
        Path(model.__file__),
        Path(human_replay.camera_geometry.__file__),
        Path(human_replay.resolution.__file__),
        Path(human_replay.measured_dynamics.__file__),
        *case_paths,
    ]
    before = [file_record(p) for p in records]
    rows = []
    for summary, path in zip(report["results"], case_paths, strict=True):
        record = json.loads(path.read_text())
        if (record["case"], record["rebound_mode"], record["status"]) != (
            summary["case"],
            summary["rebound_mode"],
            summary["status"],
        ):
            raise ValueError("case artifact/report identity mismatch")
        result = {k: record[k] for k in ("case", "rebound_mode", "status", "expected_flights")}
        if record["status"] == "fit_measured_not_quality_accepted":
            scene, heldout, bounces, _ = human_replay.load_case(
                config, cases[record["case"]], root, events
            )
            scene = replace(scene, rebound_mode=record["rebound_mode"])
            heldout = replace(heldout, rebound_mode=record["rebound_mode"])
            result.update(
                measure(
                    record,
                    scene,
                    heldout,
                    bounces,
                )
            )
            if args.render_dir:
                human_replay.render(
                    record, scene, heldout, config, root, args.render_dir / path.stem
                )
        else:
            result.update(error=record.get("error"), complete_point_accepted=False)
        rows.append(result)
    if before != [file_record(p) for p in records] or any(
        not source_identity_matches(s) for s in bindings
    ):
        raise ValueError("audit input changed")
    if args.render_dir:
        for case in cases.values():
            if (
                human_replay.frame_inventory(root / config["inputs"]["frames"] / case["clip"])
                != report["frame_inventories"][case["clip"]]
            ):
                raise ValueError("render source exposures changed")
    args.output.write_text(
        json.dumps(
            {
                "schema": "human_prefix_defect_audit_v3",
                "render_directory": str(args.render_dir) if args.render_dir else None,
                "render_outputs": [
                    {"path": str(p.resolve()), "record": file_record(p)}
                    for p in sorted(args.render_dir.iterdir())
                    if p.suffix in {".html", ".png"}
                ]
                if args.render_dir
                else [],
                "code": git_record(Path(__file__).resolve().parents[3]),
                "inputs": [
                    {"path": str(p.resolve()), "record": r}
                    for p, r in zip(records, before, strict=True)
                ],
                "scope": "Opened incomplete development prefixes; no independent 3D truth or quality certificate",
                "results": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
