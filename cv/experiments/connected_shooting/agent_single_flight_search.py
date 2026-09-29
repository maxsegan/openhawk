"""Run the real bidirectional depth beam on a frozen single-flight agent attempt.

Evaluation only. This is the additional-attempt counterpart to
``real_bidirectional_search``: original fractional event epochs are fixed,
leading-tip exposure is explicit, player/court evidence selects among a coarse
depth family, and withheld native pictures are scored only after selection.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from cv.experiments.connected_shooting import (
    agent_attempt_prepare,
    athlete_priors,
    real_bidirectional_search as whole,
    real_exposure_replay as exposure,
    serve_side,
)
from cv.pipeline import paths, provenance
from cv.validation import s6_sparse_owner_replay as replay


def server_state(
    pose_csv: Path,
    clip: str,
    frame: int,
    ball_pixel: np.ndarray,
    *,
    required_side: str | None = None,
    image_coordinate_scale: float = 1.0,
    player_name: str | None = None,
    stature_m: float | None = None,
    serve_side_association: str = "centre",
) -> dict:
    """``serve_side_association="serve_reach"`` picks an unsided first hitter by
    ``serve_side.reach_distance``; a required side is unchanged."""
    if not np.isfinite(image_coordinate_scale) or image_coordinate_scale <= 0:
        raise ValueError("positive finite player-image coordinate scale required")
    with pose_csv.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("clip") == clip and row.get("frame") == f"f_{frame:04d}.jpg"
        ]
    if not rows:
        raise ValueError("automatic player pose is absent at first post-contact picture")
    all_rows = rows
    for row in all_rows:
        row["pixel_distance"] = float(
            np.linalg.norm(
                ball_pixel
                - image_coordinate_scale
                * np.array(
                    [
                        (float(row["x0"]) + float(row["x1"])) / 2,
                        (float(row["y0"]) + float(row["y1"])) / 2,
                    ]
                )
            )
        )
    pixel_nearest = min(all_rows, key=lambda item: item["pixel_distance"])
    unscaled_pixel_nearest = min(
        all_rows,
        key=lambda item: float(
            np.linalg.norm(
                ball_pixel
                - np.array(
                    [
                        (float(item["x0"]) + float(item["x1"])) / 2,
                        (float(item["y0"]) + float(item["y1"])) / 2,
                    ]
                )
            )
        ),
    )
    if required_side is not None:
        if required_side not in {"near", "far"}:
            raise ValueError("required player side must be near or far")
        rows = [row for row in rows if row.get("side") == required_side]
        if not rows:
            raise ValueError(f"automatic {required_side}-side player pose is absent at contact")
    association = serve_side_association if required_side is None else "centre"
    row = serve_side.choose(
        rows,
        ball_pixel,
        image_coordinate_scale,
        association,
        lambda item: item["pixel_distance"],
    )
    if (player_name is None) != (stature_m is None):
        raise ValueError("player name and stature must be supplied together")
    if stature_m is not None and (not np.isfinite(stature_m) or stature_m <= 0):
        raise ValueError("positive finite player stature required")
    wrist_rows = []
    for hand in ("left_wrist", "right_wrist"):
        keys = (f"{hand}_x", f"{hand}_y", f"{hand}_confidence")
        if not all(key in row and row[key] != "" for key in keys):
            continue
        pixel = image_coordinate_scale * np.asarray(
            [float(row[keys[0]]), float(row[keys[1]])], float
        )
        confidence = float(row[keys[2]])
        if np.isfinite(pixel).all() and np.isfinite(confidence):
            wrist_rows.append(
                {
                    "hand": hand,
                    "pixel_native": pixel.tolist(),
                    "confidence": confidence,
                    "distance_to_contact_pixel_px": float(np.linalg.norm(pixel - ball_pixel)),
                }
            )
    wrist = min(wrist_rows, key=lambda item: item["distance_to_contact_pixel_px"], default=None)
    box_height = image_coordinate_scale * (float(row["y1"]) - float(row["y0"]))
    if wrist is None:
        wrist_witness = {
            "status": "abstained",
            "abstention_reason": "automatic_localization_has_no_pose_wrists",
        }
    elif box_height <= 0 or stature_m is None:
        wrist_witness = {
            **wrist,
            "status": "abstained",
            "abstention_reason": "missing_positive_stature_or_box_scale",
        }
    else:
        proxy = wrist["distance_to_contact_pixel_px"] / box_height * stature_m
        wrist_witness = {
            **wrist,
            "status": "supported" if wrist["confidence"] >= 0.25 else "abstained",
            "abstention_reason": (
                None if wrist["confidence"] >= 0.25 else "pose_wrist_confidence_below_0.25"
            ),
            "player_box_height_native_px": box_height,
            "wrist_to_ball_proxy_m": float(proxy),
            "proxy_interpretation": (
                "native wrist-to-contact pixels divided by box height and scaled by stature; "
                "an image/racket witness, not metric 3D wrist truth"
            ),
        }
    return dict(
        frame=frame,
        side=row["side"],
        player=player_name,
        stature_m=stature_m,
        court_centre_xy_m=[float(row["court_x"]), float(row["court_y"])],
        **athlete_priors.root_observation(row),
        box_height_native_px=box_height if box_height > 0 else None,
        image_association_distance_px=row["pixel_distance"],
        pixel_nearest_side=pixel_nearest["side"],
        pixel_nearest_distance_px=pixel_nearest["pixel_distance"],
        sided_file_disagrees_with_pixel_nearest=bool(pixel_nearest["side"] != row["side"]),
        unscaled_pixel_nearest_side=unscaled_pixel_nearest["side"],
        coordinate_scale_changes_pixel_nearest_side=bool(
            unscaled_pixel_nearest["side"] != pixel_nearest["side"]
        ),
        image_coordinate_scale=image_coordinate_scale,
        **(
            {
                "serve_side_association": association,
                "serve_reach_distance_heights": serve_side.reach_distance(
                    row, np.asarray(ball_pixel, float), image_coordinate_scale
                ),
            }
            if association != "centre"
            else {}
        ),
        pose_wrist_witness=wrist_witness,
        state_dof=3,
        interpretation=(
            "automatic player box court centre; nearest image player to first outgoing ball "
            + (
                "within the tennis-grammar-required alternating side; "
                if required_side is not None
                else "determines the first-contact side; "
            )
            + "racket orientation remains uncalibrated"
        ),
    )


def limits(side: str) -> tuple[float, float, np.ndarray]:
    if side == "near":
        return -0.75, 0.9144, np.arange(-1.5, 3.01, 0.5)
    if side == "far":
        return 23.77 - 0.9144, 23.77 + 0.75, np.arange(20.0, 24.51, 0.5)
    raise ValueError("near/far server side required")


def evidence(measurement: dict, player: dict, target: np.ndarray, depth_bounds, events) -> dict:
    contact = np.asarray(measurement["contact_xyz"][0])
    completion = measurement["physical"].get("terminal_completion", {})
    bounce = completion.get("end_xyz")
    bounce = None if bounce is None else np.asarray(bounce, float)
    bounce_error = None if bounce is None else float(np.linalg.norm(bounce[:2] - target[:2]))
    distance = float(np.linalg.norm(contact[:2] - player["court_centre_xy_m"]))
    direction = whole.directional_support(
        measurement["native_projection"], events, whole.SearchConfig()
    )
    checks = dict(
        connected_input_physics=bool(measurement["physical"]["compatible"]),
        serve_depth_plausible=bool(depth_bounds[0] <= contact[1] <= depth_bounds[1]),
        serve_height_plausible=bool(2.4 <= contact[2] <= 3.1),
        server_reach_plausible=bool(distance <= 1.75),
        owner_bounce_ray_agrees=bool(bounce_error is not None and bounce_error <= 0.9144),
        bidirectional_windows_supported=bool(
            direction["maximum_window_rms_px"] is not None
            and direction["maximum_window_rms_px"] <= 16
        ),
    )
    deaths = [name for name, passed in checks.items() if not passed]
    penalty = distance**2 + (1e6 if bounce_error is None else bounce_error**2)
    return dict(
        contact_xyz_m=contact.tolist(),
        player_distance_m=distance,
        bounce_xyz_m=None if bounce is None else bounce.tolist(),
        owner_bounce_ray_xyz_m=target.tolist(),
        bounce_horizontal_error_m=bounce_error,
        directional_support=direction,
        checks=checks,
        death_reasons=deaths,
        survived=not deaths,
        input_only_rank_score=float(measurement["rms_px"]["training"] ** 2 + penalty),
    )


def render(report: dict, labels: dict, output: Path) -> list[str]:
    selected = report["selected"]
    sources = agent_attempt_prepare.source_images(labels)
    clip = report["clip"]
    artifacts = []
    if selected:
        table = {row["frame"]: row for row in selected["measurement"]["native_projection"]}
        for frame in sorted(
            {int(np.ceil(report["contact_frame"])), int(report["ending_frame"])} & table.keys()
        ):
            image = Image.open(sources[(clip, frame)]).convert("RGB")
            draw = ImageDraw.Draw(image)
            row = table[frame]
            for xy, color, radius in (
                (row["owner"], "#00ff66", 8),
                (row["predicted"], "#ff3355", 6),
                (row["nominal_centre"], "#33bbff", 4),
            ):
                x, y = map(float, xy)
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius), outline=color, width=3
                )
            name = f"native_overlay_f{frame:04d}.jpg"
            image.save(output / name, quality=92)
            artifacts.append(name)
    context = report["context_frames"]
    chosen = context[:3] + context[-3:]
    thumbs = []
    for frame in chosen:
        image = Image.open(sources[(clip, frame)]).convert("RGB")
        image.thumbnail((480, 270))
        ImageDraw.Draw(image).text(
            (10, 10),
            f"native f{frame}: image context only",
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )
        thumbs.append(image)
    canvas = Image.new("RGB", (1440, 540), "black")
    for index, image in enumerate(thumbs):
        canvas.paste(image, ((index % 3) * 480, (index // 3) * 270))
    canvas.save(output / "before_after_context.jpg", quality=92)
    artifacts.append("before_after_context.jpg")
    if selected:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xyz = np.asarray(selected["measurement"]["dense_flights"][0]["positions"])
        frames = np.linspace(report["contact_frame"], report["ending_frame"], len(xyz))
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].plot(xyz[:, 0], xyz[:, 1], color="#cc3311")
        axes[0].plot([0, 10.97, 10.97, 0, 0], [0, 0, 23.77, 23.77, 0], color="black")
        axes[0].axhline(11.885, color="black", ls="--")
        axes[0].set(xlabel="court X (m)", ylabel="court Y (m)", title="court plane", aspect="equal")
        axes[1].plot(frames, xyz[:, 2], color="#cc3311")
        axes[1].axhspan(2.4, 3.1, color="green", alpha=0.12)
        axes[1].set(xlabel="native frame", ylabel="ball centre Z (m)", title="side elevation")
        fig.tight_layout()
        fig.savefig(output / "court_and_side.png", dpi=150)
        plt.close(fig)
        artifacts.append("court_and_side.png")
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("labels", "packet", "cameras", "baseline", "pose-csv", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--coarse-iterations", type=int, default=20)
    parser.add_argument("--refine-iterations", type=int, default=80)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    labels, packet, cameras, baseline = [
        json.loads(path.read_text())
        for path in (args.labels, args.packet, args.cameras, args.baseline)
    ]
    attempt = packet["attempts"][0]
    if (
        len(attempt["events"]) < 2
        or attempt["contact_count"] != 1
        or attempt["terminal_bounce_count"] != 1
    ):
        raise ValueError("this adapter requires one contact and one terminal bounce")
    scene, heldout, bounces, native = replay.prepare(attempt, cameras)
    reference = dict(annotation_status="frozen_agent_reference", records=labels["ball"]["records"])
    axes, receipt = exposure.directions(scene, bounces, [reference])
    initial = np.asarray(baseline["fit"]["parameters"], float)
    first_native = int(np.ceil(scene.contact_frames[0]))
    label = next(row for row in attempt["owner_ball_labels"] if row["frame"] == first_native)
    player = server_state(
        args.pose_csv,
        attempt["point_clip"],
        first_native,
        np.array([label["x1080"], label["y1080"]]),
    )
    low, high, depths = limits(player["side"])
    camera_table = {int(row["frame"]): np.asarray(row["P"]) for row in cameras["cameras"]}
    labels_table = {
        row["frame"]: np.array([row["x1080"], row["y1080"]])
        for row in attempt["owner_ball_labels"]
        if row["status"] == "visible"
    }
    ending_frame = int(attempt["owner_end_frame"])
    target = whole.ground_point(camera_table[ending_frame], labels_table[ending_frame])
    candidates = []
    for depth in depths:
        began = time.monotonic()
        fit = exposure.refine(
            scene,
            initial,
            bounces,
            native,
            axes,
            0.25,
            args.coarse_iterations,
            first_contact_y_m=float(depth),
        )
        measurement = exposure.measure(
            scene, heldout, bounces, native, np.asarray(fit["parameters"]), axes, 0.25, [reference]
        )
        measurement["fit"] = fit
        proof = evidence(measurement, player, target, (low, high), attempt["events"])
        candidates.append(
            dict(
                depth_hypothesis_m=float(depth),
                stage="coarse",
                measurement=measurement,
                evidence=proof,
                wall_seconds=time.monotonic() - began,
            )
        )
        print(depth, proof["survived"], measurement["rms_px"], proof["death_reasons"], flush=True)
    survivors = sorted(
        (row for row in candidates if row["evidence"]["survived"]),
        key=lambda row: row["evidence"]["input_only_rank_score"],
    )[:4]
    refined = []
    for coarse in survivors:
        fit = exposure.refine(
            scene,
            np.asarray(coarse["measurement"]["fit"]["parameters"]),
            bounces,
            native,
            axes,
            0.25,
            args.refine_iterations,
            first_contact_y_m=coarse["depth_hypothesis_m"],
        )
        measurement = exposure.measure(
            scene, heldout, bounces, native, np.asarray(fit["parameters"]), axes, 0.25, [reference]
        )
        measurement["fit"] = fit
        refined.append(
            dict(
                depth_hypothesis_m=coarse["depth_hypothesis_m"],
                stage="refined",
                measurement=measurement,
                evidence=evidence(measurement, player, target, (low, high), attempt["events"]),
            )
        )
    eligible = [row for row in refined if row["evidence"]["survived"]]
    selected = min(eligible, key=lambda row: row["evidence"]["input_only_rank_score"], default=None)
    report = dict(
        schema="s6_agent_single_flight_search_v1",
        scope=__doc__,
        status="selected_not_independently_xyz_certified"
        if selected
        else "held_no_surviving_candidate",
        attempt_id=attempt["attempt_id"],
        clip=attempt["point_clip"],
        contact_frame=float(scene.contact_frames[0]),
        ending_frame=float(scene.contact_frames[-1]),
        context_frames=attempt["context_native_frames"],
        context_is_image_only=True,
        native_event_times_changed=False,
        player_state=player,
        plausible_depth_bounds_m=[low, high],
        owner_bounce_ray_xyz_m=target.tolist(),
        observation_model=dict(duration_frames=0.25, duration_measured=False, axes=receipt),
        comparator_rms_px=baseline["euclidean_pixel_rms"],
        coarse_candidates=candidates,
        coarse_candidate_count=len(candidates),
        coarse_survivor_count=len(survivors),
        refined_candidates=refined,
        selected=selected,
        selector_uses_withheld_pixels=False,
        heldout_scored_after_selection=True,
        human_derived=True,
        automatic_inference_eligible=False,
        complete_point_accepted=False,
        inputs=[
            provenance.file_record(path)
            for path in (
                args.labels,
                args.packet,
                args.cameras,
                args.baseline,
                args.pose_csv,
                Path(__file__),
            )
        ],
        code=provenance.git_record(paths.REPO_ROOT),
    )
    args.output.mkdir(parents=True)
    report["artifacts"] = render(report, labels, args.output)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, default=replay.default, allow_nan=False) + "\n"
    )
    summary = dict(
        status=report["status"],
        coarse_candidates=len(candidates),
        coarse_survivors=len(survivors),
        selected=None
        if selected is None
        else dict(
            contact_xyz_m=selected["evidence"]["contact_xyz_m"],
            bounce_error_m=selected["evidence"]["bounce_horizontal_error_m"],
            rms_px=selected["measurement"]["rms_px"],
        ),
    )
    (args.output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><h1>Agent-labeled real S6 depth search</h1><pre>'
        + html.escape(json.dumps(summary, indent=2))
        + "</pre>"
        + "".join(
            f'<img src="{name}" style="max-width:100%;display:block">'
            for name in report["artifacts"]
        )
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
