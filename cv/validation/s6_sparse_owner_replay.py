"""Replay an entire owner-ended attempt using sparse owner pixels and events.

Evaluation only: cameras are owner-ground-conditioned, image-registered estimates.
Every fifth native labeled frame is withheld; no ball observations are filled or
discarded, and no 3D truth enters initialization, fitting, or candidate selection.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import signal
import time
import traceback

import numpy as np

from cv.experiments.connected_shooting import (
    model,
    postbounce_initialization,
    initialization,
    measured_dynamics,
    event_constraints,
    physical_compatibility,
    terminal_completion,
    regime_recovery,
    nested_rebound,
    camera_geometry,
    flight_cache,
    net_constraints,
)
from cv.pipeline import paths, provenance
from cv.validation import s6_owner_ground_camera as ground


def prepare(
    attempt: dict,
    cameras: dict,
    *,
    first_contact_offset_frames: float = 0.0,
    interior_contact_offsets_frames: dict[int, float] | None = None,
) -> tuple:
    if not np.isfinite(first_contact_offset_frames) or abs(first_contact_offset_frames) > 1:
        raise ValueError("explicit first-contact timing hypothesis must be within one frame")
    if not attempt["structurally_ground_replayable"]:
        raise ValueError("unsupported complete-attempt event grammar")
    camera_rows = cameras["cameras"]
    table = {r["frame"]: r for r in camera_rows}
    if (
        len(table) != len(camera_rows)
        or cameras["clip"] != attempt["point_clip"]
        or cameras["match_id"] != attempt["match_id"]
    ):
        raise ValueError("unique matching explicit camera inventory required")
    labels = {r["frame"]: r for r in attempt["owner_ball_labels"] if r["status"] == "visible"}
    if len(labels) != attempt["visible_native_frames"] or any(
        f not in table or table[f]["status"] != "supported" for f in labels
    ):
        raise ValueError(
            "every visible owner observation requires its supported camera; none are dropped"
        )
    bounds = np.array(
        [e["frame"] for e in attempt["events"] if e["event_type"] == "contact"]
        + [attempt["owner_end_frame"]],
        dtype=float,
    )
    # Explicit sensitivity control only. The packet's owner events remain unchanged;
    # the all-observations ownership check below rejects shifts that discard pictures.
    bounds[0] += first_contact_offset_frames
    for index, offset in (interior_contact_offsets_frames or {}).items():
        if (
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, (int, np.integer))
            or not 1 <= index < len(bounds) - 1
            or isinstance(offset, (bool, np.bool_))
            or not np.isfinite(offset)
            or abs(offset) > 1
        ):
            raise ValueError("explicit interior-contact index and offset within one frame required")
        bounds[index] += offset
    if np.any(np.diff(bounds) <= 0):
        raise ValueError("timing hypotheses must preserve strictly ordered contacts")
    train, check, native, bounces = [], [], [], []
    for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
        fs = np.array(
            sorted(f for f in labels if a <= f and (f < b or i == len(bounds) - 2 and f == b)),
            float,
        )
        native.append(fs)
        bounce = np.array(
            [
                e["frame"]
                for e in attempt["events"]
                if e["event_type"] == "bounce" and a < e["frame"] <= b
            ],
            float,
        )
        if len(bounce) != 1:
            raise ValueError("this initial replay requires one supplied bounce per flight")
        bounces.append(bounce)
        for chosen, output, minimum in ((fs[fs % 5 != 0], train, 4), (fs[fs % 5 == 0], check, 1)):
            if len(chosen) < minimum:
                raise ValueError(f"flight {i} lacks predeclared train/check coverage")
            output.append(
                (
                    chosen,
                    np.array([table[int(f)]["P"] for f in chosen]),
                    np.array([[labels[int(f)]["x1080"], labels[int(f)]["y1080"]] for f in chosen]),
                )
            )
    if sum(map(len, native)) != len(labels):
        raise ValueError("every owner observation must belong to one flight")

    def scene(rows):
        result = model.Scene(
            bounds,
            tuple(r[0] for r in rows),
            tuple(r[1] for r in rows),
            tuple(r[2] for r in rows),
            np.tile([2.0, 0.0, 0.0], (len(rows), 1)),
            attempt["fps"],
            "hard",
            "measured_240hz",
            rebound_mode="point_scales",
        )
        result.validate()
        return result

    return scene(train), scene(check), tuple(bounces), tuple(native)


def default(value):
    return value.tolist() if isinstance(value, np.ndarray) else value.item()


def plot(record: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, flight in enumerate(record["dense_flights"]):
        xyz = np.array(flight["positions"])
        axes[0].plot(xyz[:, 0], xyz[:, 1], label=str(i + 1))
        axes[1].plot(flight["query_frames"], xyz[:, 2])
    axes[0].plot([0, 10.97, 10.97, 0, 0], [0, 0, 23.77, 23.77, 0], color="gray")
    axes[0].set(
        title="Estimated overhead path; no 3D truth", xlabel="court x (m)", ylabel="court y (m)"
    )
    axes[1].set(title="Estimated ball height", xlabel="native frame", ylabel="height (m)")
    axes[1].axhline(model.R_BALL, color="gray")
    for split in ("training", "withheld"):
        rows = [r for r in record["native_projection"] if r["split"] == split]
        axes[2].scatter(
            [r["frame"] for r in rows], [r["native_error_px"] for r in rows], s=9, label=split
        )
    axes[2].set(
        title="Against original owner ball labels",
        xlabel="native frame",
        ylabel="Euclidean pixel error",
    )
    axes[2].legend()
    fig.suptitle("Sparse owner-conditioned S6 replay — not accepted 3D")
    fig.tight_layout()
    fig.savefig(output / "trajectory.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-nfev", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--recover-launch-feasibility", action="store_true")
    parser.add_argument("--memoize-flights", action="store_true")
    parser.add_argument(
        "--terminal-observation-constraint",
        action="store_true",
        help="opt-in ending guidance using the last native frame including withheld exposures",
    )
    parser.add_argument(
        "--first-contact-offset-frames",
        type=float,
        default=0.0,
        help="explicit bounded timing sensitivity, not an owner event-label correction",
    )
    parser.add_argument(
        "--net-clearance-scale-m",
        type=float,
        help="opt-in collision-free net penalty scale in metres; not an accuracy tolerance",
    )
    args = parser.parse_args()
    if args.output.exists() or min(args.max_nfev, args.timeout) <= 0:
        raise ValueError("new output and positive solve limits required")
    modules = [
        ground,
        model,
        postbounce_initialization,
        initialization,
        measured_dynamics,
        event_constraints,
        physical_compatibility,
        terminal_completion,
        regime_recovery,
        nested_rebound,
        camera_geometry,
        flight_cache,
        net_constraints,
        model.rich_ball_physics,
        model.rich_ball_physics.flight,
        model.rich_ball_physics.impact,
        measured_dynamics.bounce_reference,
        paths,
        provenance,
    ]
    files = [args.packet, args.cameras, Path(__file__)] + [Path(m.__file__) for m in modules]
    records = [provenance.file_record(p) for p in files]
    packet, cameras = (json.loads(p.read_text()) for p in (args.packet, args.cameras))
    if (
        packet.get("schema") != "s6_sparse_owner_input_packet_v1"
        or cameras.get("schema") != "s6_owner_camera_transport_v1"
        or cameras.get("human_derived") is not True
        or provenance.file_record(args.packet) not in cameras["inputs"]
    ):
        raise ValueError("source-bound owner packet and same-packet camera transport required")
    for document in (packet, cameras):
        files.extend(ground.resolve_record(r) for r in document["inputs"])
        records.extend(document["inputs"])
    attempts = [a for a in packet["attempts"] if a["point_clip"] == args.clip]
    if len(attempts) != 1:
        raise ValueError("one complete explicit owner attempt required")
    attempt = attempts[0]
    args.output.mkdir(parents=True)
    config = dict(
        max_nfev=args.max_nfev,
        timeout_seconds=args.timeout,
        training_split="frame % 5 != 0",
        withheld_split="frame % 5 == 0",
        surface="hard",
        dynamics="measured_240hz",
        rebound_mode="point_scales",
        rebound_prior_scale=0.02,
        bounce_uncertainty_frames=1.0,
        recover_bounce_regimes=True,
        bounce_regime_strategy="fixed_branch",
        warm_start_rebound=True,
        retain_direct_rebound=True,
        initialization="existing post-bounce fallback when incoming training frames <4",
        recover_launch_feasibility=args.recover_launch_feasibility,
        memoize_flights=args.memoize_flights,
        net_clearance_scale_m=args.net_clearance_scale_m,
        first_contact_offset_frames=args.first_contact_offset_frames,
        original_first_contact_frame=attempt["first_event_frame"],
        owner_event_labels_changed=False,
        terminal_observation_constraint=args.terminal_observation_constraint,
    )
    record = dict(
        schema="s6_sparse_owner_replay_v1",
        scope=__doc__,
        human_derived=True,
        inputs=records,
        configuration=config,
        code=provenance.git_record(paths.REPO_ROOT),
        attempt_id=attempt["attempt_id"],
        status="failed",
        complete_point_accepted=False,
        independent_3d_accuracy_measured=False,
        observations_discarded=0,
    )
    (args.output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    started = time.monotonic()

    def timed_out(_signum, frame):
        record["timeout_stack"] = [
            dict(file=Path(row.filename).name, function=row.name, line=row.lineno)
            for row in traceback.extract_stack(frame)
        ]
        raise TimeoutError("whole-attempt fitting deadline")

    prior = signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(args.timeout)
    try:
        scene, heldout, bounces, native = prepare(
            attempt, cameras, first_contact_offset_frames=args.first_contact_offset_frames
        )
        record["coverage"] = dict(
            flights=len(native),
            native_owner_observations=sum(map(len, native)),
            training=sum(map(len, scene.observation_frames)),
            withheld=sum(map(len, heldout.observation_frames)),
            native_frames_by_flight=native,
        )
        print("Initializing sparse owner attempt", flush=True)
        seed, evidence = postbounce_initialization.seed(
            scene,
            [float(b[0]) for b in bounces],
            max_nfev=args.max_nfev,
            recover_launch_feasibility=args.recover_launch_feasibility,
        )
        seed = np.r_[seed, 1.0, 1.0]
        record.update(initial_parameters=seed, initialization=evidence)
        (args.output / "initialization.json").write_text(
            json.dumps(dict(seed=seed, evidence=evidence), default=default, indent=2) + "\n"
        )
        print("Fitting connected complete attempt", flush=True)
        result = model.fit(
            scene,
            seed,
            max_nfev=args.max_nfev,
            optimize_spin=True,
            bounce_frames=bounces,
            bounce_uncertainty_frames=1.0,
            recover_bounce_regimes=True,
            bounce_regime_strategy="fixed_branch",
            rebound_prior_scale=0.02,
            warm_start_rebound=True,
            retain_direct_rebound=True,
            memoize_flights=args.memoize_flights,
            net_clearance_scale_m=args.net_clearance_scale_m,
            **(
                {"terminal_last_observation_frame": float(native[-1][-1])}
                if args.terminal_observation_constraint
                else {}
            ),
        )
        record.update(status="fit_measured_not_quality_accepted", fit=result)
        signal.alarm(0)
        projections = []
        for split, target in (("training", scene), ("withheld", heldout)):
            for i, flight in enumerate(model.chain(target, result["parameters"])):
                predicted = camera_geometry.project(target.cameras[i], flight["positions"])
                for frame, actual, estimate in zip(
                    target.observation_frames[i], target.pixels[i], predicted, strict=True
                ):
                    projections.append(
                        dict(
                            frame=int(frame),
                            flight=i,
                            split=split,
                            owner_xy=actual,
                            fitted_xy=estimate,
                            native_error_px=float(np.linalg.norm(actual - estimate)),
                        )
                    )
        record["native_projection"] = sorted(projections, key=lambda r: r["frame"])
        record["euclidean_pixel_rms"] = {
            s: float(
                np.sqrt(
                    np.mean([r["native_error_px"] ** 2 for r in projections if r["split"] == s])
                )
            )
            for s in ("training", "withheld")
        }
        kind = "second_bounce" if attempt["terminal_bounce_count"] == 2 else "terminal_bounce"
        record["physical_compatibility"] = physical_compatibility.evaluate(
            scene, result["parameters"], bounces, kind, native
        )
        queries = tuple(
            np.unique(np.r_[np.arange(a, b, scene.fps / 240), b])
            for a, b in zip(scene.contact_frames, scene.contact_frames[1:])
        )
        dense = model.chain(scene, result["parameters"], query_frames=queries)
        record["dense_flights"] = [
            {**r, "query_frames": f} for r, f in zip(dense, queries, strict=True)
        ]
        plot(record, args.output)
    except (ValueError, KeyError, TimeoutError, FloatingPointError, OverflowError) as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prior)
    record["wall_seconds"] = time.monotonic() - started
    if records != [provenance.file_record(p) for p in files]:
        raise ValueError("replay inputs changed")
    (args.output / "report.json").write_text(
        json.dumps(record, default=default, indent=2, allow_nan=False) + "\n"
    )
    summary = {
        k: record[k]
        for k in (
            "status",
            "error",
            "coverage",
            "euclidean_pixel_rms",
            "physical_compatibility",
            "wall_seconds",
        )
        if k in record
    }
    (args.output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>S6 sparse owner replay</title><h1>Sparse owner-conditioned replay — not accepted 3D</h1>'
        + ('<img src="trajectory.png" style="max-width:100%">' if "dense_flights" in record else "")
        + "<pre>"
        + html.escape(json.dumps(summary, default=default, indent=2))
        + "</pre>"
    )
    print(
        json.dumps(
            {
                k: record[k]
                for k in ("status", "error", "euclidean_pixel_rms", "wall_seconds")
                if k in record
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
