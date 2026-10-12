"""Refine camera vertical scale from registered player height without moving the court plane.

The canonical net-calibrated camera remains the baseline. This automatic stage jointly selects a
bijective player-to-side assignment and one bounded vertical-column scale per point using grounded
pose observations. A refinement is accepted only when player reprojection improves and the observed
net-cord residual does not regress. Ground-plane projection is unchanged by construction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from cv.pipeline.camera_cal import NET_POST_X, NET_Y, net_height_at_x, project
from cv.pipeline.run_manifest import StageRun

HEAD_HEIGHT_FRACTION = 0.945
MINIMUM_OBSERVATIONS = 8
MAXIMUM_HELD_PLAYER_RMS_PX = 20.0
SCALE_BOUNDS = (0.85, 1.15)


def frame_number(value: str | int) -> int:
    return int(value) if isinstance(value, int) else int(Path(value).stem.rsplit("_", 1)[-1])


def scaled_vertical_projection(projection: np.ndarray, scale: float) -> np.ndarray:
    refined = np.asarray(projection, dtype=float).copy()
    refined[:, 2] *= float(scale)
    return refined


def projected_net_residual(projection: np.ndarray, observed: np.ndarray) -> float | None:
    observed = np.asarray(observed, dtype=float).reshape(-1, 2)
    observed = observed[np.isfinite(observed).all(axis=1)]
    if len(observed) < 2:
        return None
    court_x = np.linspace(NET_POST_X[0], NET_POST_X[1], 161)
    physical = project(
        projection,
        np.stack(
            (
                court_x,
                np.full_like(court_x, NET_Y),
                [net_height_at_x(float(value)) for value in court_x],
            ),
            axis=1,
        ),
    )
    distances = np.sqrt(
        np.min(np.sum((observed[:, None, :] - physical[None, :, :]) ** 2, axis=2), axis=1)
    )
    return float(np.sqrt(np.mean(np.square(distances))))


def load_pose_observations(path: Path) -> tuple[dict[tuple[str, str], list[dict]], dict]:
    coordinates = json.loads(Path(f"{path}.coordinates.json").read_text())
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            confidence = min(
                float(row.get("nose_confidence", 0.0)),
                float(row.get("left_ankle_confidence", 0.0)),
                float(row.get("right_ankle_confidence", 0.0)),
            )
            if row.get("side") not in {"near", "far"} or confidence < 0.40:
                continue
            foot_y = 0.5 * (float(row["left_ankle_y"]) + float(row["right_ankle_y"]))
            box_height = max(float(row["y1"]) - float(row["y0"]), 1.0)
            if float(row["y1"]) - foot_y > 0.14 * box_height:
                continue
            grouped[(row["clip"], row["side"])].append(
                {
                    "frame": frame_number(row["frame"]),
                    "root_xy": np.asarray([float(row["court_x"]), float(row["court_y"])]),
                    "head_xy": np.asarray([float(row["nose_x"]), float(row["nose_y"])]),
                    "confidence": confidence,
                    "pixel_height": max(0.0, foot_y - float(row["nose_y"])),
                }
            )
    for observations in grouped.values():
        if len(observations) < MINIMUM_OBSERVATIONS:
            continue
        cutoff = float(np.quantile([row["pixel_height"] for row in observations], 0.60))
        observations[:] = [row for row in observations if row["pixel_height"] >= cutoff]
    return grouped, coordinates


def player_residuals(
    observations: list[dict],
    projections: dict[int, np.ndarray],
    height_m: float,
    scale: float,
) -> np.ndarray:
    residuals = []
    for row in observations:
        projection = projections.get(row["frame"])
        if projection is None:
            continue
        point = [*row["root_xy"], HEAD_HEIGHT_FRACTION * height_m]
        predicted = project(scaled_vertical_projection(projection, scale), [point])[0]
        residuals.append((predicted[1] - row["head_xy"][1]) * math.sqrt(row["confidence"]))
    return np.asarray(residuals, dtype=float)


def robust_cost(residuals: np.ndarray, delta: float = 8.0) -> float:
    absolute = np.abs(residuals)
    return float(np.mean(np.where(absolute <= delta, 0.5 * residuals**2, delta * absolute)))


def fit_clip(
    clip: str,
    observations: dict[tuple[str, str], list[dict]],
    projections: dict[int, np.ndarray],
    net_cord: np.ndarray | None,
    players: list[dict],
) -> dict:
    sides = [side for side in ("near", "far") if observations.get((clip, side))]
    if not sides or not players:
        return {"accepted": False, "reason": "missing_player_height_observations"}
    fit_observations = {
        side: [row for row in observations[(clip, side)] if row["frame"] % 2 == 0] for side in sides
    }
    evaluation_observations = {
        side: [row for row in observations[(clip, side)] if row["frame"] % 2 == 1] for side in sides
    }
    if any(
        len(fit_observations[side]) < 4 or len(evaluation_observations[side]) < 4 for side in sides
    ):
        return {"accepted": False, "reason": "insufficient_held_frame_height_observations"}
    candidates = []
    for ordered in permutations(players, min(len(sides), len(players))):
        if len(ordered) != len(sides):
            continue
        assignment = dict(zip(sides, ordered, strict=True))

        def objective(scale: float) -> float:
            residuals = np.concatenate(
                [
                    player_residuals(
                        fit_observations[side],
                        projections,
                        float(assignment[side]["height_m"]),
                        scale,
                    )
                    for side in sides
                ]
            )
            player_cost = robust_cost(residuals)
            if net_cord is None:
                net_cost = 0.0
            else:
                reference = projections[next(iter(projections))]
                net_error = projected_net_residual(
                    scaled_vertical_projection(reference, scale), net_cord
                )
                net_cost = 0.5 * (net_error or 0.0) ** 2
            prior_cost = 4.0 * ((scale - 1.0) / 0.10) ** 2
            return player_cost + net_cost + prior_cost

        result = minimize_scalar(
            objective,
            bounds=SCALE_BOUNDS,
            method="bounded",
            options={"xatol": 1e-4, "maxiter": 50},
        )
        scale = float(result.x)
        before = np.concatenate(
            [
                player_residuals(
                    evaluation_observations[side],
                    projections,
                    float(assignment[side]["height_m"]),
                    1.0,
                )
                for side in sides
            ]
        )
        after = np.concatenate(
            [
                player_residuals(
                    evaluation_observations[side],
                    projections,
                    float(assignment[side]["height_m"]),
                    scale,
                )
                for side in sides
            ]
        )
        reference = projections[next(iter(projections))]
        candidates.append(
            {
                "objective": float(result.fun),
                "scale": scale,
                "assignment": {
                    side: {
                        "player_id": assignment[side]["player_id"],
                        "height_m": float(assignment[side]["height_m"]),
                        "body_profile": assignment[side]["body_profile"],
                    }
                    for side in sides
                },
                "fit_observations": int(sum(len(fit_observations[side]) for side in sides)),
                "held_observations": int(len(after)),
                "player_rms_before_px": float(np.sqrt(np.mean(before**2))),
                "player_rms_after_px": float(np.sqrt(np.mean(after**2))),
                "net_rms_before_px": projected_net_residual(reference, net_cord)
                if net_cord is not None
                else None,
                "net_rms_after_px": projected_net_residual(
                    scaled_vertical_projection(reference, scale), net_cord
                )
                if net_cord is not None
                else None,
            }
        )
    if not candidates:
        return {"accepted": False, "reason": "no_bijective_player_assignment"}
    candidates.sort(key=lambda row: row["objective"])
    best = candidates[0]
    margin = candidates[1]["objective"] - best["objective"] if len(candidates) > 1 else math.inf
    dimension_span = max(float(row["height_m"]) for row in players) - min(
        float(row["height_m"]) for row in players
    )
    identity_safe = margin >= 1.0 or len(players) == 1
    assignment_safe = margin >= 1.0 or dimension_span <= 0.03 or len(players) == 1
    net_safe = best["net_rms_after_px"] is None or best["net_rms_after_px"] <= max(
        float(best["net_rms_before_px"]) + 1.5,
        1.15 * float(best["net_rms_before_px"]),
    )
    improves_player = best["player_rms_after_px"] + 1.0 <= best["player_rms_before_px"]
    best.update(
        {
            "schema": "metric_camera_clip_refinement_v1",
            "clip": clip,
            "assignment_margin": None if not math.isfinite(margin) else float(margin),
            "identity_safe": identity_safe,
            "assignment_safe": assignment_safe,
            "accepted": bool(
                best["held_observations"] >= MINIMUM_OBSERVATIONS
                and assignment_safe
                and net_safe
                and improves_player
                and best["player_rms_after_px"] <= MAXIMUM_HELD_PLAYER_RMS_PX
            ),
            "reason": (
                "accepted_player_height_net_consistent"
                if best["held_observations"] >= MINIMUM_OBSERVATIONS
                and assignment_safe
                and net_safe
                and improves_player
                and best["player_rms_after_px"] <= MAXIMUM_HELD_PLAYER_RMS_PX
                else "abstained_no_safe_metric_improvement"
            ),
        }
    )
    return best


def refine_match(
    match_id: str,
    camera_path: Path,
    pose_path: Path,
    biometrics_path: Path,
    output_path: Path,
    dimensions_path: Path,
) -> dict:
    biometrics = json.loads(biometrics_path.read_text())
    players = [
        {
            "player_id": player_id,
            "height_m": float(biometrics["players"][player_id]["height_m"]),
            "body_profile": (
                "female_smpl"
                if biometrics["players"][player_id].get("sex") == "female"
                else "male_smpl"
                if biometrics["players"][player_id].get("sex") == "male"
                else "female_smpl"
                if "wtatennis.com" in biometrics["players"][player_id].get("source", "")
                else "male_smpl"
                if "atptour.com" in biometrics["players"][player_id].get("source", "")
                else "neutral"
            ),
            "source": biometrics["players"][player_id].get("source"),
        }
        for player_id in biometrics.get("matches", {}).get(match_id, [])
        if player_id in biometrics.get("players", {})
    ]
    observations, _coordinates = load_pose_observations(pose_path)
    with np.load(camera_path, allow_pickle=True) as source:
        arrays = {name: source[name].copy() for name in source.files}
    by_clip = defaultdict(dict)
    for clip, frame, projection in zip(arrays["clips"], arrays["frames"], arrays["P"], strict=True):
        by_clip[str(clip)][int(frame)] = np.asarray(projection, dtype=float)
    net_by_clip = {}
    if "net_cord_xy" in arrays and "net_cord_valid" in arrays:
        for index, clip in enumerate(arrays["clips"]):
            if str(clip) not in net_by_clip and bool(arrays["net_cord_valid"][index]):
                net_by_clip[str(clip)] = np.asarray(arrays["net_cord_xy"][index], dtype=float)
    refinements = {
        clip: fit_clip(clip, observations, projections, net_by_clip.get(clip), players)
        for clip, projections in sorted(by_clip.items())
    }
    metric_refined = np.zeros(len(arrays["P"]), dtype=bool)
    metric_scale = np.ones(len(arrays["P"]), dtype=float)
    player_residual = np.full(len(arrays["P"]), np.nan, dtype=float)
    for index, clip_value in enumerate(arrays["clips"]):
        refinement = refinements[str(clip_value)]
        if refinement.get("accepted"):
            arrays["P"][index] = scaled_vertical_projection(arrays["P"][index], refinement["scale"])
            metric_refined[index] = True
            metric_scale[index] = refinement["scale"]
            player_residual[index] = refinement["player_rms_after_px"]
    arrays["metric_refined"] = metric_refined
    arrays["metric_scale"] = metric_scale
    arrays["player_height_residual_px"] = player_residual
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    dimensions = {
        "schema": "player_body_dimensions_v1",
        "automatic_only": True,
        "human_derived_inputs": [],
        "match_id": match_id,
        "players": players,
        "clip_side_assignments": {
            f"{clip}/{side}": {
                **assignment,
                "scope": "clip_side",
                "source": "metric_camera_registered_height_assignment",
                "assignment_safe": bool(row.get("assignment_safe")),
                "identity_safe": bool(row.get("identity_safe")),
                "assignment_margin": row.get("assignment_margin"),
            }
            for clip, row in refinements.items()
            for side, assignment in row.get("assignment", {}).items()
        },
        "refinements": refinements,
    }
    dimensions_path.write_text(json.dumps(dimensions, indent=2, sort_keys=True) + "\n")
    return {
        "clips": len(refinements),
        "refined": sum(bool(row.get("accepted")) for row in refinements.values()),
        "output": str(output_path),
        "dimensions": str(dimensions_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-dir", type=Path, required=True)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--camera-name", default="camera_P_per_frame_v1.npz")
    parser.add_argument("--pose-name", default="player_pose_tracked_crop_native_v1.csv")
    parser.add_argument("--output-name", default="camera_P_metric_v1.npz")
    parser.add_argument("--dimensions-name", default="player_body_dimensions_v1.json")
    parser.add_argument(
        "--biometrics",
        type=Path,
        default=Path(__file__).with_name("player_biometrics.json"),
    )
    args = parser.parse_args()
    stage = StageRun(
        args.match_dir,
        "camera_metric_refine",
        args,
        reused_artifacts=[
            str(args.match_dir / args.camera_name),
            str(args.match_dir / args.pose_name),
            str(args.biometrics),
        ],
    )
    report = refine_match(
        args.match_id,
        args.match_dir / args.camera_name,
        args.match_dir / args.pose_name,
        args.biometrics,
        args.match_dir / args.output_name,
        args.match_dir / args.dimensions_name,
    )
    stage.finish(outputs=report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
