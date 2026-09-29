"""Coarse-to-fine bidirectional whole-point search for opened real S6 attempts.

The search consumes explicit human ball/event evidence and an owner-conditioned
camera, so it is evaluation-only.  It enumerates serve depth on the first native
observation ray, fits every candidate with the same connected 6DoF-per-flight
state and bounded spin nuisance, scores short/medium windows on both sides of
each physical event, applies coarse player/serve/net/bounce evidence, and refines
only the surviving whole-point beam.  Held-out pictures are scored only after
selection.  Pre-contact and post-ending pictures are rendered as context and do
not become observations or change native event times.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import html
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from cv.experiments.connected_shooting import real_exposure_replay as exposure
from cv.experiments.connected_shooting import serve_speed_witness
from cv.pipeline import paths, provenance
from cv.validation import s6_sparse_owner_replay as replay

FAR_BASELINE_Y_M = 23.77
BALL_RADIUS_M = 0.0325


@dataclass(frozen=True)
class SearchConfig:
    depth_min_m: float = 20.0
    depth_max_m: float = 24.5
    depth_step_m: float = 0.5
    coarse_iterations: int = 20
    refine_iterations: int = 100
    beam_width: int = 6
    exposure_duration_frames: float = 0.25
    short_horizon_frames: int = 4
    medium_horizon_frames: int = 10
    serve_inside_allowance_m: float = 0.9144
    serve_behind_allowance_m: float = 0.75
    serve_height_min_m: float = 2.4
    serve_height_max_m: float = 3.1
    player_reach_m: float = 1.75
    bounce_tolerance_m: float = 0.9144
    directional_rms_limit_px: float = 16.0

    def validate(self) -> None:
        finite = [
            self.depth_min_m,
            self.depth_max_m,
            self.depth_step_m,
            self.exposure_duration_frames,
            self.serve_inside_allowance_m,
            self.serve_behind_allowance_m,
            self.serve_height_min_m,
            self.serve_height_max_m,
            self.player_reach_m,
            self.bounce_tolerance_m,
            self.directional_rms_limit_px,
        ]
        if not np.isfinite(finite).all() or self.depth_step_m <= 0:
            raise ValueError("finite search ranges and positive depth step required")
        if not self.depth_min_m < self.depth_max_m:
            raise ValueError("ordered nonempty depth range required")
        if not 0 < self.exposure_duration_frames <= 1:
            raise ValueError("bounded positive exposure duration required")
        if self.serve_height_min_m >= self.serve_height_max_m:
            raise ValueError("ordered serve-height prior required")
        for value in (
            self.coarse_iterations,
            self.refine_iterations,
            self.beam_width,
            self.short_horizon_frames,
            self.medium_horizon_frames,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("positive integer search budgets required")


def ray_point_at_y(camera: np.ndarray, pixel: np.ndarray, court_y_m: float) -> np.ndarray:
    """Intersect a native pinhole observation ray with a constant court-Y plane."""
    camera, pixel = np.asarray(camera, float), np.asarray(pixel, float)
    if camera.shape != (3, 4) or pixel.shape != (2,) or not np.isfinite(camera).all():
        raise ValueError("finite pinhole camera and one native pixel required")
    planes = camera[:2] - pixel[:, None] * camera[2]
    augmented = np.vstack([planes, [0.0, 1.0, 0.0, -court_y_m]])
    _, _, right = np.linalg.svd(augmented)
    point = right[-1]
    if abs(point[3]) < 1e-12:
        raise ValueError("observation ray is parallel to requested court-Y plane")
    xyz = point[:3] / point[3]
    if not np.isfinite(xyz).all():
        raise ValueError("nonfinite observation-ray intersection")
    return xyz


def ground_point(
    camera: np.ndarray, pixel: np.ndarray, radial: np.ndarray | None = None
) -> np.ndarray:
    """Project an event click to the ball-centre ground plane."""
    camera, pixel = np.asarray(camera, float), np.asarray(pixel, float)
    if radial is not None:
        from cv.experiments.connected_shooting import camera_geometry

        pixel = camera_geometry.undistort(pixel[None], np.asarray(radial, float)[None])[0]
    planes = camera[:2] - pixel[:, None] * camera[2]
    augmented = np.vstack([planes, [0.0, 0.0, 1.0, -BALL_RADIUS_M]])
    _, _, right = np.linalg.svd(augmented)
    point = right[-1]
    if abs(point[3]) < 1e-12:
        raise ValueError("observation ray does not meet the ground plane")
    return point[:3] / point[3]


def load_player_states(pose_csv: Path, clip: str, contacts: tuple[int, ...]) -> list[dict]:
    """Return coarse 3DoF player states: court centre and image arm-face proxy."""
    with pose_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    states = []
    sides = ("far", "near")
    for frame, side in zip(contacts, sides, strict=True):
        selected = [
            row
            for row in rows
            if row.get("clip") == clip
            and row.get("frame") == f"f_{frame:04d}.jpg"
            and row.get("side") == side
        ]
        if len(selected) != 1:
            raise ValueError(f"one {side} player pose required at frame {frame}")
        row = selected[0]
        shoulder = np.array(
            [
                np.mean([float(row["left_shoulder_x"]), float(row["right_shoulder_x"])]),
                np.mean([float(row["left_shoulder_y"]), float(row["right_shoulder_y"])]),
            ]
        )
        wrists = [
            (
                float(row[f"{hand}_wrist_confidence"]),
                np.array([float(row[f"{hand}_wrist_x"]), float(row[f"{hand}_wrist_y"])]),
            )
            for hand in ("left", "right")
        ]
        confidence, wrist = max(wrists, key=lambda item: item[0])
        states.append(
            dict(
                frame=frame,
                side=side,
                court_centre_xy_m=[float(row["court_x"]), float(row["court_y"])],
                racket_face_proxy_angle_rad=float(math.atan2(*(wrist - shoulder)[::-1])),
                proxy_wrist_confidence=confidence,
                state_dof=3,
                interpretation="automatic box ground centre plus shoulder-to-wrist image angle; not metric racket-face truth",
            )
        )
    return states


def rms(values: list[float]) -> float | None:
    return float(np.sqrt(np.mean(np.square(values)))) if values else None


def directional_support(
    native_projection: list[dict], events: list[dict], cfg: SearchConfig
) -> dict:
    """Score short/medium observed windows approaching and leaving each event.

    These pixels have already been generated by the connected forward physical
    propagation. Grouping its residuals from an event outward in both temporal
    directions is the coarse bidirectional message used by the beam.
    """
    training = [row for row in native_projection if row["split"] == "training"]
    windows = []
    for event in events:
        frame = float(event["frame"])
        for horizon_name, horizon in (
            ("short", cfg.short_horizon_frames),
            ("medium", cfg.medium_horizon_frames),
        ):
            for direction, predicate in (
                ("backward", lambda value: frame - horizon <= value <= frame),
                ("forward", lambda value: frame <= value <= frame + horizon),
            ):
                errors = [row["error_px"] for row in training if predicate(float(row["frame"]))]
                windows.append(
                    dict(
                        event_type=event["event_type"],
                        event_frame=frame,
                        direction=direction,
                        horizon=horizon_name,
                        native_training_pictures=len(errors),
                        rms_px=rms(errors),
                    )
                )
    measured = [row["rms_px"] for row in windows if row["rms_px"] is not None]
    return dict(
        windows=windows,
        maximum_window_rms_px=max(measured) if measured else None,
        propagation="connected rich-physics candidate, grouped from each event in both directions",
    )


def candidate_evidence(
    measurement: dict,
    attempt: dict,
    player_states: list[dict],
    bounce_targets: list[np.ndarray],
    cfg: SearchConfig,
) -> dict:
    contact = np.asarray(measurement["contact_xyz"][0], float)
    return_contact = np.asarray(measurement["contact_xyz"][1], float)
    serve_player = np.asarray(player_states[0]["court_centre_xy_m"], float)
    return_player = np.asarray(player_states[1]["court_centre_xy_m"], float)
    modeled_bounces = [
        (
            np.asarray(measurement["dense_flights"][0]["bounces"][0]["x"], float)
            if measurement["dense_flights"][0]["bounces"]
            else None
        ),
        (
            np.asarray(measurement["dense_flights"][1]["bounces"][0]["x"], float)
            if measurement["dense_flights"][1]["bounces"]
            else np.asarray(
                measurement["physical"].get("terminal_completion", {}).get("end_xyz"),
                float,
            )
            if measurement["physical"].get("terminal_completion", {}).get("end_xyz") is not None
            else None
        ),
    ]
    bounce_errors = [
        None if actual is None else float(np.linalg.norm(actual[:2] - target[:2]))
        for actual, target in zip(modeled_bounces, bounce_targets, strict=True)
    ]
    direction = directional_support(measurement["native_projection"], attempt["events"], cfg)
    checks = dict(
        optimizer_converged=bool(measurement["fit"]["success"]),
        connected_input_physics=bool(measurement["physical"]["compatible"]),
        serve_depth_plausible=bool(
            FAR_BASELINE_Y_M - cfg.serve_inside_allowance_m
            <= contact[1]
            <= FAR_BASELINE_Y_M + cfg.serve_behind_allowance_m
        ),
        serve_height_plausible=bool(cfg.serve_height_min_m <= contact[2] <= cfg.serve_height_max_m),
        server_reach_plausible=bool(
            np.linalg.norm(contact[:2] - serve_player) <= cfg.player_reach_m
        ),
        returner_reach_plausible=bool(
            np.linalg.norm(return_contact[:2] - return_player) <= cfg.player_reach_m
        ),
        owner_bounce_rays_agree=bool(
            all(error is not None and error <= cfg.bounce_tolerance_m for error in bounce_errors)
        ),
        bidirectional_windows_supported=bool(
            direction["maximum_window_rms_px"] is not None
            and direction["maximum_window_rms_px"] <= cfg.directional_rms_limit_px
        ),
    )
    gates = (
        "connected_input_physics",
        "serve_depth_plausible",
        "serve_height_plausible",
        "server_reach_plausible",
        "returner_reach_plausible",
        "owner_bounce_rays_agree",
        "bidirectional_windows_supported",
    )
    deaths = [name for name in gates if not checks[name]]
    # Selection is deliberately truth-free and excludes withheld pixels.
    training_rms = measurement["rms_px"]["training"]
    player_distance = float(np.linalg.norm(contact[:2] - serve_player))
    prior_penalty = (
        4.0 * max(FAR_BASELINE_Y_M - cfg.serve_inside_allowance_m - contact[1], 0.0) ** 2
        + 4.0 * max(contact[1] - FAR_BASELINE_Y_M - cfg.serve_behind_allowance_m, 0.0) ** 2
        + max(cfg.serve_height_min_m - contact[2], 0.0) ** 2
        + max(contact[2] - cfg.serve_height_max_m, 0.0) ** 2
        + player_distance**2
        + sum(1e6 if error is None else error**2 for error in bounce_errors)
    )
    return dict(
        contact_xyz_m=contact.tolist(),
        return_contact_xyz_m=return_contact.tolist(),
        serve_player_distance_m=player_distance,
        return_player_distance_m=float(np.linalg.norm(return_contact[:2] - return_player)),
        bounce_xyz_m=[None if row is None else row.tolist() for row in modeled_bounces],
        owner_bounce_ray_xyz_m=[row.tolist() for row in bounce_targets],
        bounce_horizontal_errors_m=bounce_errors,
        directional_support=direction,
        checks=checks,
        death_reasons=deaths,
        survived=not deaths,
        input_only_rank_score=float(training_rms**2 + prior_penalty),
        selector_uses_withheld_pixels=False,
    )


def draw_native_overlays(
    output: Path, frames_root: Path, rows: list[dict], selected: dict
) -> list[str]:
    chosen = {284, 298, 309, 335}
    lookup = {int(row["frame"]): row for row in selected["measurement"]["native_projection"]}
    written = []
    for frame in sorted(chosen & lookup.keys()):
        source = frames_root / f"pt0002_{frame:04d}.jpg"
        image = Image.open(source).convert("RGB")
        draw = ImageDraw.Draw(image)
        row = lookup[frame]
        for xy, color, radius in (
            (row["owner"], "#00ff66", 8),
            (row["predicted"], "#ff3355", 6),
            (row["nominal_centre"], "#33bbff", 4),
        ):
            x, y = map(float, xy)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
        draw.text(
            (25, 25),
            f"native f{frame}  green=owner front  red=exposure prediction  blue=centre",
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )
        name = f"native_overlay_f{frame:04d}.jpg"
        image.save(output / name, quality=92)
        written.append(name)
    return written


def context_montage(output: Path, frames_root: Path) -> str:
    frames = (280, 282, 283, 336, 338, 339)
    thumbs = []
    for frame in frames:
        image = Image.open(frames_root / f"pt0002_{frame:04d}.jpg").convert("RGB")
        image.thumbnail((480, 270))
        draw = ImageDraw.Draw(image)
        phase = "pre-contact context" if frame < 284 else "post-ending context"
        draw.text((10, 10), f"f{frame}: {phase}", fill="white", stroke_width=2, stroke_fill="black")
        thumbs.append(image)
    canvas = Image.new("RGB", (1440, 540), "black")
    for index, image in enumerate(thumbs):
        canvas.paste(image, ((index % 3) * 480, (index // 3) * 270))
    name = "before_after_context.jpg"
    canvas.save(output / name, quality=92)
    return name


def plot_geometry(output: Path, candidates: list[dict], selected: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#999999" if not row["evidence"]["survived"] else "#4477aa" for row in candidates]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(
        [row["evidence"]["contact_xyz_m"][1] for row in candidates],
        [row["measurement"]["rms_px"]["training"] for row in candidates],
        c=colors,
        label="coarse depth family",
    )
    ax.scatter(
        [selected["evidence"]["contact_xyz_m"][1]],
        [selected["measurement"]["rms_px"]["training"]],
        marker="*",
        s=180,
        color="#cc3311",
        label="selected/refined",
    )
    ax.axvspan(
        FAR_BASELINE_Y_M - 0.9144,
        FAR_BASELINE_Y_M + 0.75,
        alpha=0.14,
        color="green",
        label="plausible serve depth",
    )
    ax.set(
        xlabel="serve contact court Y (m)",
        ylabel="training RMS (px)",
        title="Real observation-ray depth family",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "depth_family.png", dpi=150)
    plt.close(fig)

    flights = selected["measurement"]["dense_flights"]
    xyz = np.concatenate([np.asarray(f["positions"]) for f in flights])
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.plot(xyz[:, 0], xyz[:, 1], color="#cc3311", lw=2)
    ax.scatter(
        *np.asarray(selected["evidence"]["bounce_xyz_m"])[:, :2].T,
        marker="x",
        s=70,
        label="modeled bounces",
    )
    ax.scatter(
        *np.asarray(selected["evidence"]["owner_bounce_ray_xyz_m"])[:, :2].T,
        facecolors="none",
        edgecolors="#009988",
        s=90,
        label="owner bounce-frame rays",
    )
    ax.scatter(
        *np.asarray(selected["evidence"]["contact_xyz_m"])[:2],
        marker="*",
        s=140,
        label="serve contact",
    )
    ax.plot([0, 10.97, 10.97, 0, 0], [0, 0, 23.77, 23.77, 0], color="black", lw=1)
    ax.axhline(11.885, color="black", ls="--", lw=1)
    ax.set(
        xlim=(-1, 12),
        ylim=(-2, 26),
        aspect="equal",
        xlabel="court X (m)",
        ylabel="court Y (m)",
        title="Selected connected whole point — court plane",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "court_plane.png", dpi=150)
    plt.close(fig)

    frames = np.concatenate(
        [np.linspace(f["start_frame"], f["end_frame"], len(f["positions"])) for f in flights]
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(frames, xyz[:, 2], color="#cc3311", lw=2)
    for event in (284, 298, 309, 335):
        ax.axvline(event, color="#777777", ls=":" if event not in (284, 309) else "--")
    ax.axhspan(2.4, 3.1, color="green", alpha=0.12, label="serve-height prior")
    ax.set(
        xlabel="native frame",
        ylabel="ball centre Z (m)",
        title="Selected connected whole point — side elevation",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "side_elevation.png", dpi=150)
    plt.close(fig)
    return ["depth_family.png", "court_plane.png", "side_elevation.png"]


def run(args: argparse.Namespace) -> dict:
    cfg = SearchConfig(
        depth_min_m=args.depth_min_m,
        depth_max_m=args.depth_max_m,
        depth_step_m=args.depth_step_m,
        coarse_iterations=args.coarse_iterations,
        refine_iterations=args.refine_iterations,
        beam_width=args.beam_width,
    )
    cfg.validate()
    packet, cameras, exposure_report = [
        json.loads(path.read_text()) for path in (args.packet, args.cameras, args.exposure_report)
    ]
    speed_document = json.loads(args.serve_speed.read_text()) if args.serve_speed else None
    speed_reading = None if speed_document is None else speed_document["reading"]
    if (
        len(packet.get("attempts", [])) != 1
        or exposure_report.get("schema") != "s6_real_exposure_replay_v1"
    ):
        raise ValueError("one explicit attempt and completed real exposure replay required")
    if (
        exposure_report.get("status") != "complete"
        or exposure_report["results"]["front_d025"]["status"] != "measured"
    ):
        raise ValueError("completed 0.25-frame real exposure condition required")
    attempt = packet["attempts"][0]
    scene, heldout, bounces, native = replay.prepare(
        attempt, cameras, first_contact_offset_frames=args.first_contact_offset_frames
    )
    references = [json.loads(path.read_text()) for path in args.streak_reference]
    axes, axis_receipt = exposure.directions(scene, bounces, references)
    baseline_parameters = np.asarray(
        exposure_report["results"]["front_d025"]["fit"]["parameters"], float
    )
    player_states = load_player_states(args.pose_csv, attempt["point_clip"], (284, 309))
    camera_lookup = {int(row["frame"]): np.asarray(row["P"], float) for row in cameras["cameras"]}
    owner_lookup = {
        int(row["frame"]): np.asarray([row["x1080"], row["y1080"]], float)
        for row in attempt["owner_ball_labels"]
    }
    bounce_targets = [
        ground_point(camera_lookup[frame], owner_lookup[frame]) for frame in (298, 335)
    ]
    depth_values = np.arange(
        cfg.depth_min_m, cfg.depth_max_m + 0.5 * cfg.depth_step_m, cfg.depth_step_m
    )
    ray_family = [
        dict(
            court_y_m=float(depth),
            xyz_m=ray_point_at_y(camera_lookup[284], owner_lookup[284], float(depth)).tolist(),
        )
        for depth in depth_values
    ]
    candidates = []
    for depth in depth_values:
        began = time.monotonic()
        fit = exposure.refine(
            scene,
            baseline_parameters,
            bounces,
            native,
            axes,
            cfg.exposure_duration_frames,
            cfg.coarse_iterations,
            first_contact_y_m=float(depth),
        )
        measurement = exposure.measure(
            scene,
            heldout,
            bounces,
            native,
            np.asarray(fit["parameters"]),
            axes,
            cfg.exposure_duration_frames,
            references,
        )
        measurement["fit"] = fit
        evidence = candidate_evidence(measurement, attempt, player_states, bounce_targets, cfg)
        candidate = dict(
            depth_hypothesis_m=float(depth),
            stage="coarse",
            measurement=measurement,
            evidence=evidence,
            wall_seconds=time.monotonic() - began,
        )
        evidence["survived_before_serve_speed"] = evidence["survived"]
        if speed_reading is not None:
            serve_speed_witness.add_to_candidate(candidate, speed_reading)
        candidates.append(candidate)
        print(
            depth,
            evidence["survived"],
            measurement["rms_px"],
            evidence["death_reasons"],
            flush=True,
        )

    survivors = [row for row in candidates if row["evidence"]["survived"]]
    survivors.sort(
        key=lambda row: (row["evidence"]["input_only_rank_score"], row["depth_hypothesis_m"])
    )
    survivors = survivors[: cfg.beam_width]
    refined = []
    for coarse in survivors:
        began = time.monotonic()
        fit = exposure.refine(
            scene,
            np.asarray(coarse["measurement"]["fit"]["parameters"]),
            bounces,
            native,
            axes,
            cfg.exposure_duration_frames,
            cfg.refine_iterations,
            first_contact_y_m=coarse["depth_hypothesis_m"],
        )
        measurement = exposure.measure(
            scene,
            heldout,
            bounces,
            native,
            np.asarray(fit["parameters"]),
            axes,
            cfg.exposure_duration_frames,
            references,
        )
        measurement["fit"] = fit
        evidence = candidate_evidence(measurement, attempt, player_states, bounce_targets, cfg)
        candidate = dict(
            depth_hypothesis_m=coarse["depth_hypothesis_m"],
            stage="refined",
            measurement=measurement,
            evidence=evidence,
            wall_seconds=time.monotonic() - began,
        )
        evidence["survived_before_serve_speed"] = evidence["survived"]
        if speed_reading is not None:
            serve_speed_witness.add_to_candidate(candidate, speed_reading)
        refined.append(candidate)
    eligible = [row for row in refined if row["evidence"]["survived"]]
    selected = min(
        eligible,
        key=lambda row: (row["evidence"]["input_only_rank_score"], row["depth_hypothesis_m"]),
        default=None,
    )
    files = [
        args.packet,
        args.cameras,
        args.exposure_report,
        args.pose_csv,
        args.frames_root,
        Path(__file__),
        Path(exposure.__file__),
        *args.streak_reference,
    ]
    if args.serve_speed is not None:
        files.append(args.serve_speed)
    bindings = []
    for path in files:
        if path.is_dir():
            continue
        bindings.append(provenance.file_record(path))
    report = dict(
        schema="s6_real_bidirectional_search_v1",
        scope=__doc__,
        status="selected_not_independently_xyz_certified"
        if selected
        else "held_no_surviving_candidate",
        attempt_id=attempt["attempt_id"],
        configuration=cfg.__dict__,
        inputs=bindings,
        code=provenance.git_record(paths.REPO_ROOT),
        human_derived=True,
        automatic_inference_eligible=False,
        observation_model=dict(
            duration_frames=cfg.exposure_duration_frames,
            opening_offset_frames=0.0,
            blur_radius_px=1.5,
            duration_measured=False,
            axes=axis_receipt,
        ),
        first_observation_ray_family=ray_family,
        player_states=player_states,
        owner_bounce_ray_xyz_m=[row.tolist() for row in bounce_targets],
        coarse_candidates=candidates,
        coarse_candidate_count=len(candidates),
        coarse_survivor_count_before_serve_speed=sum(
            row["evidence"]["survived_before_serve_speed"] for row in candidates
        ),
        coarse_survivor_count=len(survivors),
        refined_candidates=refined,
        selected=selected,
        selector_uses_withheld_pixels=False,
        heldout_scored_after_selection=True,
        context=dict(
            pre_contact_frames=[280, 282, 283],
            post_ending_frames=[336, 338, 339],
            image_context_only=True,
            native_event_times_changed=False,
            invented_pictures=0,
        ),
        complete_point_accepted=False,
        independent_xyz_truth_available=False,
        serve_speed_graphic=None if speed_document is None else speed_document,
        first_contact_timing_witness=dict(
            original_frame=float(attempt["first_event_frame"]),
            fitted_frame=float(scene.contact_frames[0]),
            offset_frames=float(args.first_contact_offset_frames),
        ),
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("packet", "cameras", "exposure-report", "pose-csv", "frames-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--serve-speed", type=Path)
    parser.add_argument("--streak-reference", type=Path, nargs="+", required=True)
    parser.add_argument("--coarse-iterations", type=int, default=20)
    parser.add_argument("--refine-iterations", type=int, default=100)
    parser.add_argument("--beam-width", type=int, default=6)
    parser.add_argument("--depth-min-m", type=float, default=20.0)
    parser.add_argument("--depth-max-m", type=float, default=24.5)
    parser.add_argument("--depth-step-m", type=float, default=0.5)
    parser.add_argument("--first-contact-offset-frames", type=float, default=0.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report = run(args)
    selected = report["selected"]
    artifacts = []
    if selected:
        artifacts += draw_native_overlays(
            args.output,
            args.frames_root,
            report["selected"]["measurement"]["native_projection"],
            selected,
        )
        artifacts += plot_geometry(args.output, report["coarse_candidates"], selected)
    artifacts.append(context_montage(args.output, args.frames_root))
    report["artifacts"] = artifacts
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, default=replay.default, allow_nan=False) + "\n"
    )
    summary = dict(
        status=report["status"],
        coarse_candidates=report["coarse_candidate_count"],
        coarse_survivors=report["coarse_survivor_count"],
        selected=None
        if selected is None
        else dict(
            contact_xyz_m=selected["evidence"]["contact_xyz_m"],
            bounce_errors_m=selected["evidence"]["bounce_horizontal_errors_m"],
            rms_px=selected["measurement"]["rms_px"],
            input_only_rank_score=selected["evidence"]["input_only_rank_score"],
        ),
    )
    (args.output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Real bidirectional S6 search</title><h1>Real connected whole-point depth search</h1><p>Opened human-derived evaluation input; not automatic inference or independent XYZ certification. Held-out pixels were scored only after selection.</p><pre>'
        + html.escape(json.dumps(summary, indent=2))
        + "</pre>"
        + "".join(
            f'<img src="{name}" style="max-width:100%;display:block;margin:1rem 0">'
            for name in artifacts
        )
        + '<p><a href="report.json">Complete candidate evidence and provenance</a></p>'
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
