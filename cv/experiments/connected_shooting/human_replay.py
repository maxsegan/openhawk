"""Can connected shooting replay selected real-video human-event prefixes?

Opened development only. Human times and manual prefix selection condition the
fit; cameras/tracks remain automatic and unverified. No 3D truth or old quality
verdict enters fitting. Incomplete prefixes can never qualify complete points.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import html
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

from cv.experiments.connected_shooting import (
    initialization,
    measured_dynamics,
    model,
    event_constraints,
    camera_geometry,
)
from cv.pipeline.provenance import file_record, git_record
from cv.pipeline import resolution


def frame_inventory(path: Path) -> dict:
    files = sorted(path.glob("f_*.jpg"))
    digest = hashlib.sha256()
    frames = []
    for file in files:
        frames.append(int(file.stem.removeprefix("f_")))
        digest.update(file.name.encode())
        digest.update(hashlib.sha256(file.read_bytes()).digest())
    return {"count": len(files), "sha256": digest.hexdigest(), "frames": frames}


def effective_config(
    config: dict,
    candidate_top1: str | Path | None,
    camera_input: str | Path | None = None,
) -> dict:
    result = {**config, "inputs": dict(config["inputs"])}
    if candidate_top1 is not None:
        result["inputs"]["track"] = str(candidate_top1)
        result["observation_mode"] = "candidate_top1"
    if camera_input is not None:
        result["inputs"]["cameras"] = str(camera_input)
    return result


def load_case(config: dict, case: dict, root: Path, events_path: Path) -> tuple:
    """Keep native exposure ownership disjoint; do not repair retained input tracks."""
    lo, hi = map(float, case["contact_bounds"])
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError("ordered contact-prefix bounds required")
    key = config["match_id"] + "__" + case["clip"]
    with events_path.open(newline="") as handle:
        physical = [
            r
            for r in csv.DictReader(handle)
            if r["clip"] == key
            and r["verdict"] in {"confirmed", "adjusted", "new"}
            and r["event_type"] in {"contact", "bounce", "net_hit"}
        ]
    selected = sorted(
        [r for r in physical if lo <= float(r["labeled_frame"]) <= hi],
        key=lambda r: float(r["labeled_frame"]),
    )
    contacts = np.array(
        [float(r["labeled_frame"]) for r in selected if r["event_type"] == "contact"]
    )
    if (
        len(contacts) != case["expected_flights"] + 1
        or contacts[0] != lo
        or contacts[-1] != hi
        or np.any(np.diff(contacts) <= 0)
    ):
        raise ValueError("prefix must be bounded by the declared distinct human contacts")
    if any(r["event_type"] == "net_hit" for r in selected):
        raise ValueError("net-impact prefixes are unsupported, not silently split")
    bounces = []
    for a, b in zip(contacts, contacts[1:]):
        between = [
            float(r["labeled_frame"])
            for r in selected
            if r["event_type"] == "bounce" and a < float(r["labeled_frame"]) < b
        ]
        if len(between) != 1:
            raise ValueError("this fixed replay requires one reviewed bounce per contact interval")
        bounces.append(between[0])
    scene, heldout, evidence = load_observations(config, case["clip"], root, contacts)
    return (
        scene,
        heldout,
        bounces,
        {
            **evidence,
            "human_events": selected,
            "physical_events_outside_prefix": len(physical) - len(selected),
        },
    )


def load_observations(config: dict, clip: str, root: Path, boundaries: np.ndarray) -> tuple:
    """Load immutable native evidence for explicit flight boundaries, including a terminal end.

    The final boundary is a sampling horizon, not necessarily a racket contact.
    Event selection/meaning belongs to the caller. No missing observations are filled.
    """
    contacts = np.asarray(boundaries, float)
    if (
        contacts.ndim != 1
        or len(contacts) < 2
        or not np.isfinite(contacts).all()
        or np.any(np.diff(contacts) <= 0)
    ):
        raise ValueError("finite ordered flight boundaries required")
    lo, hi = contacts[[0, -1]]
    track_path = root / config["inputs"]["track"]
    coordinates = json.loads(
        track_path.with_name(track_path.name + ".coordinates.json").read_text()
    )
    frame_coordinates = json.loads(
        (root / (config["inputs"]["frames"] + ".coordinates.json")).read_text()
    )
    native = np.array(config["native_size"], int)
    for metadata in [coordinates, frame_coordinates]:
        if [metadata["image_size"][k] for k in ("width", "height")] != native.tolist():
            raise ValueError("explicit native image sizes disagree")
    if config["camera_pixel_space"] != "native" or frame_coordinates["fps"] != config["fps"]:
        raise ValueError("explicit camera pixel space and native cadence required")
    observations = {}
    observation_mode = config.get("observation_mode", "retained_track")
    if observation_mode not in {"retained_track", "candidate_top1"}:
        raise ValueError("explicit supported observation mode required")
    with track_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns, coordinate_size = resolution.coordinate_columns_and_size(
            reader.fieldnames or [],
            coordinates,
            native_columns=("x_native", "y_native"),
            legacy_columns=("x", "y"),
        )
        artifact = np.array([coordinate_size.width, coordinate_size.height], float)
        if np.any(artifact <= 0) or not np.isfinite(artifact).all():
            raise ValueError("finite positive track coordinate size required")
        scale = native / artifact
        for row in reader:
            if row["clip"] != clip:
                continue
            if observation_mode == "candidate_top1" and int(row["rank"]) != 0:
                continue
            frame = int(Path(row["frame"]).stem.removeprefix("f_"))
            if not lo <= frame <= hi:
                continue
            xy = np.array([float(row[column]) for column in columns]) * scale
            if (
                frame in observations
                or not np.isfinite(xy).all()
                or np.any(xy < 0)
                or np.any(xy >= native)
            ):
                raise ValueError("unique finite in-image retained ball observations required")
            observations[frame] = xy
    cameras, camera_evidence, radial_by_frame = {}, [], {}
    with np.load(root / config["inputs"]["cameras"], allow_pickle=False) as source:
        has_radial = "k1" in source or "dist_center" in source
        if has_radial and (
            "k1" not in source
            or "dist_center" not in source
            or source["k1"].shape != (len(source["frames"]),)
            or source["dist_center"].shape != (len(source["frames"]), 2)
        ):
            raise ValueError("complete per-frame native radial-camera metadata required")
        if "k2" in source and (not np.isfinite(source["k2"]).all() or np.any(source["k2"] != 0)):
            raise ValueError("second-order radial cameras are unsupported")
        for index in np.flatnonzero(source["clips"].astype(str) == clip):
            frame = int(source["frames"][index])
            if frame not in observations:
                continue
            if (
                frame in cameras
                or not bool(source["reliable"][index])
                or not np.isfinite(source["P"][index]).all()
            ):
                raise ValueError("duplicate, held or nonfinite automatic camera")
            cameras[frame] = source["P"][index]
            if has_radial:
                radial_by_frame[frame] = np.r_[source["k1"][index], source["dist_center"][index]]
            camera_evidence.append(
                {
                    "frame": frame,
                    **{
                        k: source[k][index].item()
                        for k in (
                            "source",
                            "confidence",
                            "fallback_ancestry",
                            "frame_scope",
                            "reference_frame",
                        )
                        if k in source
                    },
                }
            )
    groups = [[], [], [], [], [], []]
    radial_groups = [[], []]
    for i, (a, b) in enumerate(zip(contacts, contacts[1:])):
        # An integer contact exposure belongs to the outgoing interval once;
        # the prefix's final contact belongs to its final incoming interval.
        frames = np.array(
            sorted(
                f for f in observations if a <= f and (f < b or i == len(contacts) - 2 and f == b)
            ),
            float,
        )
        train, test = frames[frames % 3 != 0], frames[frames % 3 == 0]
        if len(train) < 4 or len(test) < 1:
            raise ValueError("insufficient retained native training/heldout observations")
        for chosen, offset in [(train, 0), (test, 3)]:
            groups[offset].append(chosen)
            groups[offset + 1].append(np.stack([cameras[int(f)] for f in chosen]))
            groups[offset + 2].append(np.stack([observations[int(f)] for f in chosen]))
            if has_radial:
                radial_groups[offset // 3].append(
                    np.stack([radial_by_frame[int(f)] for f in chosen])
                )
    scene = model.Scene(
        contacts,
        tuple(groups[0]),
        tuple(groups[1]),
        tuple(groups[2]),
        np.tile([2.0, 0.0, 0.0], (len(contacts) - 1, 1)),
        config["fps"],
        config["surface"],
        "measured_240hz",
        camera_distortion=tuple(radial_groups[0]) if has_radial else None,
    )
    heldout = replace(
        scene,
        observation_frames=tuple(groups[3]),
        cameras=tuple(groups[4]),
        pixels=tuple(groups[5]),
        camera_distortion=tuple(radial_groups[1]) if has_radial else None,
    )
    scene.validate()
    heldout.validate()
    return (
        scene,
        heldout,
        {
            "retained_observations": len(observations),
            "observation_mode": observation_mode,
            "missing_native_frames": [
                f for f in range(int(np.ceil(lo)), int(np.floor(hi)) + 1) if f not in observations
            ],
            "camera_evidence": camera_evidence,
            "camera_projection": "native_radial_k1" if has_radial else "native_pinhole",
            "track_to_native_scale": scale.tolist(),
            "track_coordinate_columns": list(columns),
            "track_coordinate_size": artifact.tolist(),
            "event_coordinate_provenance": "CSV does not distinguish track-prefilled coordinates from manual clicks",
            "independent_coordinate_clicks_verified": False,
            "camera_status": "automatic estimates, not independently certified metric truth",
        },
    )


def json_default(value):
    return value.tolist() if isinstance(value, np.ndarray) else value.item()


def native_observations(scene: model.Scene, heldout: model.Scene, parameters) -> dict:
    """One original observation and lens-aware fit projection per native exposure."""
    observations = {}
    for target in (scene, heldout):
        fitted = model.chain(target, parameters)
        for i, (fs, ps, cameras, fit) in enumerate(
            zip(target.observation_frames, target.pixels, target.cameras, fitted, strict=True)
        ):
            radial = None if target.camera_distortion is None else target.camera_distortion[i]
            projected = camera_geometry.project(cameras, fit["positions"], radial)
            for frame, automatic, prediction in zip(fs, ps, projected, strict=True):
                if frame in observations:
                    raise ValueError("duplicated native exposure ownership")
                observations[frame] = (automatic, prediction)
    return observations


def render(
    record: dict, scene: model.Scene, heldout: model.Scene, config: dict, root: Path, output: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parameters = np.asarray(record["parameters"])
    scope_label = record.get("scope_label", "human-conditioned INCOMPLETE prefix")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for flight in record["dense_flights"]:
        xyz = np.asarray(flight["positions"])
        axes[0].plot(xyz[:, 0], xyz[:, 1])
        axes[1].plot(xyz[:, 1], xyz[:, 2])
    for target, label in [(scene, "training"), (heldout, "heldout")]:
        residual = model.image_residual(target, parameters).reshape(-1, 2)
        axes[2].scatter(
            np.concatenate(target.observation_frames),
            np.linalg.norm(residual, axis=1),
            s=8,
            label=label,
        )
    axes[0].plot([0, 10.97, 10.97, 0, 0], [0, 0, 23.77, 23.77, 0], color="gray")
    axes[0].set(
        xlabel="court x (m)", ylabel="court y (m)", title="Estimated overhead path; not 3D truth"
    )
    axes[1].axhline(model.R_BALL, color="gray")
    axes[1].set(xlabel="court y (m)", ylabel="height (m)", title="Estimated side view")
    axes[2].set(
        xlabel="native frame",
        ylabel="native pixel error",
        title="Against automatic tracks, not truth",
    )
    axes[2].legend()
    fig.suptitle(
        f"{record['case']} / {record['rebound_mode']} / "
        f"{'bounce-window objective' if record.get('fit_bounce_windows') else 'image-only objective'}"
        f" — {scope_label}; no quality verdict"
    )
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=120)
    plt.close(fig)
    # Original native JPEGs are referenced unchanged; SVG markers are a separate overlay.
    panels = []
    projected_observations = native_observations(scene, heldout, parameters)
    for event in record["evidence"]["human_events"]:
        labeled_frame = float(event["labeled_frame"])
        frame = int(np.ceil(labeled_frame))
        image = root / config["inputs"]["frames"] / record["clip"] / f"f_{frame:04d}.jpg"
        if not image.exists():
            continue
        url = html.escape(os.path.relpath(image, output.parent))
        archived_overlay = ""
        if all(event.get(k) not in {None, ""} for k in ("labeled_x540", "labeled_y540")):
            x = float(event["labeled_x540"]) * config["native_size"][0] / 960
            y = float(event["labeled_y540"]) * config["native_size"][1] / 540
            archived_overlay = (
                f'<circle cx="{x}" cy="{y}" r="12" fill="none" stroke="cyan" stroke-width="3"/>'
            )
        prediction_overlay = ""
        prediction_note = "No supported native observation for a fit overlay on this exposure."
        if frame in projected_observations:
            observed, predicted = projected_observations[frame]
            ox, oy = observed
            px, py = predicted
            prediction_overlay = (
                f'<circle cx="{ox}" cy="{oy}" r="6" fill="none" stroke="orange" stroke-width="3"/>'
                f'<path d="M {px - 9} {py - 9} L {px + 9} {py + 9} M {px - 9} {py + 9} L {px + 9} {py - 9}" stroke="magenta" stroke-width="3"/>'
            )
            prediction_note = (
                "Fit projection and original automatic observation at this native exposure."
            )
        panels.append(
            f'<h3>{html.escape(event["event_type"])}: labeled time {labeled_frame:g}, displayed native frame {frame}</h3><p>{prediction_note} Archived coordinates, when supplied, belong to the labeled time, not necessarily this exposure. <a href="{url}">Open original native image</a></p><div style="position:relative"><img loading="lazy" src="{url}" style="width:100%;display:block"><svg viewBox="0 0 {config["native_size"][0]} {config["native_size"][1]}" style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none">{archived_overlay}{prediction_overlay}</svg></div>'
        )
    output.with_suffix(".html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Human-conditioned Stage 6 diagnostic</title><main style="max-width:1200px;margin:auto;font-family:sans-serif"><h1>'
        + html.escape(scope_label)
        + ' — not accepted 3D</h1><p>Human event times; automatic cameras and ball tracks. Cyan markers are archived event coordinates, which may have been prefilled from tracks; independent manual clicks are not verified. No old 3D verdict applies to this fit.</p><img style="width:100%" src="'
        + output.with_suffix(".png").name
        + '">'
        + "<p>Magenta X: fitted 3D projected through the declared lens model. Orange ring: original automatic track. Cyan ring: archived event coordinate (not independently verified).</p>"
        + "".join(panels)
        + "</main>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--match-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rebound-modes", nargs="+", choices=["fixed", "point_scales"], default=["fixed"]
    )
    parser.add_argument("--max-nfev", type=int, default=60)
    parser.add_argument("--point-timeout", type=int, default=180)
    parser.add_argument("--consensus-initialization", action="store_true")
    parser.add_argument("--fit-bounce-windows", action="store_true")
    parser.add_argument("--include-bounce-exposure", action="store_true")
    parser.add_argument(
        "--camera-input",
        type=Path,
        help="explicit native per-frame camera NPZ control; never replaces the source archive",
    )
    parser.add_argument(
        "--candidate-top1",
        type=Path,
        help="explicit detector candidate CSV: use every rank-0 observation instead of the retained track",
    )
    args = parser.parse_args()
    if min(args.max_nfev, args.point_timeout) <= 0 or len(set(args.rebound_modes)) != len(
        args.rebound_modes
    ):
        parser.error("positive limits and unique modes required")
    config = effective_config(
        json.loads(args.cases.read_text()), args.candidate_top1, args.camera_input
    )
    if file_record(args.events)["sha256"] != config["event_labels_sha256"]:
        raise ValueError("explicit archived event corpus hash mismatch")
    sources = [
        args.cases,
        args.events,
        Path(__file__),
        Path(model.__file__),
        Path(camera_geometry.__file__),
        Path(resolution.__file__),
        Path(event_constraints.__file__),
        Path(initialization.__file__),
        Path(measured_dynamics.__file__),
        Path(measured_dynamics.bounce_reference.__file__),
        Path(model.rich_ball_physics.__file__),
        Path(model.rich_ball_physics.flight.__file__),
        Path(model.rich_ball_physics.impact.__file__),
    ]
    sources.extend(args.match_root / config["inputs"][k] for k in ("track", "cameras"))
    sources += [
        args.match_root / (config["inputs"]["track"] + ".coordinates.json"),
        args.match_root / (config["inputs"]["frames"] + ".coordinates.json"),
    ]
    inputs = [file_record(p) for p in sources]
    inventories = {}
    for case in config["cases"]:
        inventory = frame_inventory(args.match_root / config["inputs"]["frames"] / case["clip"])
        if (
            inventory["count"] != case["frame_count"]
            or inventory["sha256"] != case["frame_inventory_sha256"]
        ):
            raise ValueError("source exposure inventory differs from verified human-label images")
        inventories[case["clip"]] = inventory
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "connected_human_prefix_replay_v1",
        "status": "running",
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "code": git_record(Path(__file__).resolve().parents[3]),
        "inputs": [
            {"resolved_path": str(p.resolve()), "record": r}
            for p, r in zip(sources, inputs, strict=True)
        ],
        "frame_inventories": inventories,
        "scope": config["scope"],
        "observation_mode": config.get("observation_mode", "retained_track"),
        "observation_source": str((args.match_root / config["inputs"]["track"]).resolve()),
        "camera_source": str((args.match_root / config["inputs"]["cameras"]).resolve()),
        "complete_points_accepted": 0,
        "current_fit_human_quality_labels": 0,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    for case in config["cases"]:
        for mode in args.rebound_modes:
            started = time.monotonic()
            record = {
                "case": config["match_id"] + "__" + case["clip"],
                "clip": case["clip"],
                "rebound_mode": mode,
                "fit_bounce_windows": args.fit_bounce_windows,
                "expected_flights": case["expected_flights"],
                "status": "failed",
                "complete_point_accepted": False,
                "independent_3d_accuracy_measured": False,
                "human_quality_on_this_fit": None,
            }

            def timed_out(*_):
                raise TimeoutError("human prefix fitting limit")

            previous = signal.signal(signal.SIGALRM, timed_out)
            signal.alarm(args.point_timeout)
            try:
                scene, heldout, bounces, evidence = load_case(
                    config, case, args.match_root, args.events
                )
                scene, heldout = (
                    replace(scene, rebound_mode=mode),
                    replace(heldout, rebound_mode=mode),
                )
                seed, seed_evidence = initialization.physics_backward_seed(
                    scene,
                    bounces,
                    consensus=args.consensus_initialization,
                    include_bounce_exposure=args.include_bounce_exposure,
                )
                if mode == "point_scales":
                    seed = np.r_[seed, 1.0, 1.0]
                record.update(
                    evidence=evidence, initialization=seed_evidence, initial_parameters=seed
                )
                result = model.fit(
                    scene,
                    seed,
                    optimize_spin=True,
                    max_nfev=args.max_nfev,
                    bounce_frames=tuple(np.array([frame]) for frame in bounces)
                    if args.fit_bounce_windows
                    else None,
                )
                queries = tuple(
                    np.unique(np.r_[np.arange(a, b, scene.fps / 240), b])
                    for a, b in zip(scene.contact_frames, scene.contact_frames[1:])
                )
                record.update(
                    status="fit_measured_not_quality_accepted",
                    parameters=result["parameters"],
                    optimizer_success=result["optimizer_success"],
                    objective_calls=result["objective_calls"],
                    bounce_window_evidence=result["bounce_window_evidence"],
                    training_pixel_rms=result["final_pixel_rms"],
                    heldout_pixel_rms=float(
                        np.sqrt(np.mean(model.image_residual(heldout, result["parameters"]) ** 2))
                    ),
                    dense_flights=model.chain(scene, result["parameters"], query_frames=queries),
                )
                signal.alarm(0)
                render(
                    record,
                    scene,
                    heldout,
                    config,
                    args.match_root,
                    args.output / f"{case['clip']}_{mode}",
                )
            except (
                ValueError,
                KeyError,
                FileNotFoundError,
                TimeoutError,
                FloatingPointError,
            ) as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous)
            record["wall_seconds"] = time.monotonic() - started
            rows.append(record)
            (args.output / f"{case['clip']}_{mode}.json").write_text(
                json.dumps(record, default=json_default, indent=2) + "\n"
            )
            print(
                json.dumps(
                    {
                        k: record.get(k)
                        for k in (
                            "case",
                            "rebound_mode",
                            "status",
                            "heldout_pixel_rms",
                            "error",
                            "wall_seconds",
                        )
                    }
                ),
                flush=True,
            )
    if inputs != [file_record(p) for p in sources] or any(
        frame_inventory(args.match_root / config["inputs"]["frames"] / clip) != value
        for clip, value in inventories.items()
    ):
        raise ValueError("human replay inputs changed")
    manifest.update(
        status="complete",
        attempted_prefixes=len(rows),
        fit_measured_prefixes=sum(r["status"] == "fit_measured_not_quality_accepted" for r in rows),
        results=[
            {
                k: r.get(k)
                for k in (
                    "case",
                    "rebound_mode",
                    "status",
                    "expected_flights",
                    "heldout_pixel_rms",
                    "error",
                )
            }
            for r in rows
        ],
    )
    (args.output / "report.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
