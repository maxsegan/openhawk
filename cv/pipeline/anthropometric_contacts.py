"""Anthropometric 3D contact volumes from one camera and a known tennis court.

The ground homography fixes a confidently grounded foot in court coordinates.  Known player
height plus the observed head-to-foot image scale estimates the ball/racket contact height in
the player's local projective scale.  Intersecting the observed contact ray with that height
plane gives a 3D center, capped by a human reach envelope around the last grounded foot.

This is a detector-blind geometry prior, not a contact detector and not a hard point.  Near
and far contacts carry different covariance, pose/ball timing failures inflate it, and the
downstream chain solver is free to move within the volume.  Tier-1 labels are never read.

Example:
  .venv/bin/python cv/pipeline/anthropometric_contacts.py \
      --match rg2025f --clips pt0092
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import rich_ball_physics as rich  # noqa: E402
import resolution as res  # noqa: E402
from run_manifest import StageRun  # noqa: E402

PLAYER_HEIGHT_M = {
    "Jannik_Sinner": 1.91,
    "Carlos_Alcaraz": 1.83,
}
DEFAULT_HEIGHT_M = 1.87
HEAD_HEIGHT_FRACTION = 0.94  # COCO nose/eye centroid, not skull top
GROUND_WINDOW_F = 14
POSE_WINDOW_F = 4
BALL_WINDOW_F = 7

POSE_KEYS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


@dataclass
class PoseFrame:
    frame: int
    side: str
    box: np.ndarray
    conf: float
    points: dict[str, tuple[np.ndarray, float]]

    @property
    def height_px(self) -> float:
        return float(self.box[3] - self.box[1])


def parse_frame(raw: str) -> int:
    return int(raw[2:6]) if raw.startswith("f_") else int(round(float(raw)))


def load_pose_file(path: str, clip: str, scale: float) -> dict[int, PoseFrame]:
    poses = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["clip"] != clip:
                continue
            points = {}
            for key in POSE_KEYS:
                points[key] = (
                    scale * np.array([float(row[f"{key}_x"]), float(row[f"{key}_y"])]),
                    float(row[f"{key}_confidence"]),
                )
            frame = parse_frame(row["frame"])
            poses[frame] = PoseFrame(
                frame=frame, side=row["side"],
                box=scale * np.array([float(row[k]) for k in ("x0", "y0", "x1", "y1")]),
                conf=float(row["conf"]), points=points,
            )
    return poses


def nearest_pose(poses: dict[int, PoseFrame], frame: float, window: int = POSE_WINDOW_F):
    if not poses:
        return None
    best = min(poses, key=lambda value: abs(value - frame))
    return poses[best] if abs(best - frame) <= window else None


def valid_points(pose: PoseFrame, keys, threshold=0.25):
    return [point for key in keys for point, conf in [pose.points[key]]
            if conf >= threshold and np.all(point > 0)]


def foot_pixel(pose: PoseFrame):
    ankles = valid_points(pose, ("left_ankle", "right_ankle"), 0.25)
    if ankles:
        # The lower ankle is the grounded foot during a split-step; average only ankles
        # within 6% of body height so the airborne trailing foot does not move the anchor.
        lower = max(point[1] for point in ankles)
        grounded = [point for point in ankles if lower - point[1] <= 0.06 * pose.height_px]
        return np.mean(grounded, axis=0), "ankle"
    return np.array([(pose.box[0] + pose.box[2]) / 2, pose.box[3]]), "box_bottom"


def is_grounded(pose: PoseFrame):
    foot, method = foot_pixel(pose)
    tolerance = 0.14 * max(pose.height_px, 1.0)
    return method == "ankle" and pose.box[3] - foot[1] <= tolerance


def last_grounded_pose(poses: dict[int, PoseFrame], frame: float):
    candidates = [pose for f, pose in poses.items() if frame - GROUND_WINDOW_F <= f <= frame]
    grounded = [pose for pose in candidates if is_grounded(pose)]
    if grounded:
        return max(grounded, key=lambda pose: pose.frame), True
    pose = nearest_pose(poses, frame, GROUND_WINDOW_F)
    return pose, False


def head_pixel(pose: PoseFrame):
    eyes_nose = valid_points(pose, ("nose", "left_eye", "right_eye"), 0.3)
    if eyes_nose:
        return np.mean(eyes_nose, axis=0), "face"
    shoulders = valid_points(pose, ("left_shoulder", "right_shoulder"), 0.3)
    if shoulders:
        shoulder = np.mean(shoulders, axis=0)
        return shoulder + np.array([0.0, -0.22 * pose.height_px]), "shoulder_extrapolation"
    return np.array([(pose.box[0] + pose.box[2]) / 2, pose.box[1]]), "box_top"


def racket_pixel(pose: PoseFrame, ball_uv: np.ndarray | None):
    candidates = []
    for side in ("left", "right"):
        wrist, wc = pose.points[f"{side}_wrist"]
        elbow, ec = pose.points[f"{side}_elbow"]
        if min(wc, ec) < 0.25 or not np.all(wrist > 0) or not np.all(elbow > 0):
            continue
        # Adult forearm ~0.45 m, racket center-to-sweet-spot ~0.50 m. The image-vector
        # extension is perspective-local and therefore more reliable than a world yaw guess.
        sweet_spot = wrist + 1.1 * (wrist - elbow)
        distance = float(np.linalg.norm(sweet_spot - ball_uv)) if ball_uv is not None else -wrist[1]
        candidates.append((distance, sweet_spot, wrist, min(wc, ec), side))
    return min(candidates, key=lambda value: value[0]) if candidates else None


def ray_at_z(P: np.ndarray, uv: np.ndarray, z: float):
    u, v = uv
    A = np.array([
        [P[0, 0] - u * P[2, 0], P[0, 1] - u * P[2, 1]],
        [P[1, 0] - v * P[2, 0], P[1, 1] - v * P[2, 1]],
    ])
    b = -np.array([
        P[0, 2] * z + P[0, 3] - u * (P[2, 2] * z + P[2, 3]),
        P[1, 2] * z + P[1, 3] - v * (P[2, 2] * z + P[2, 3]),
    ])
    try:
        xy = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return np.array([xy[0], xy[1], z])


def load_identities(out: str):
    import json

    path = os.path.join(out, "side_identity_v1.json")
    if not os.path.exists(path):
        return {}, {}
    with open(path) as handle:
        data = json.load(handle)
    return data.get("near_cluster", {}), data.get("cluster_names", {})


def player_for_side(clip: str, side: str, near_cluster, cluster_names):
    point = str(int(clip[2:]))
    near = near_cluster.get(point)
    if near is None:
        return "unknown"
    cluster = near if side == "near" else ({"A", "B"} - {near}).pop()
    return cluster_names.get(cluster, cluster)


def nearest_ball(ball: dict[int, np.ndarray], frame: float):
    candidates = [f for f in ball if abs(f - frame) <= BALL_WINDOW_F]
    if not candidates:
        return None, None
    best = min(candidates, key=lambda f: abs(f - frame))
    return ball[best], best


def contact_volume(contact, poses, ball, camera: rich.FrameCamera, height_m: float):
    frame = contact["frame"]
    pose = nearest_pose(poses, frame)
    grounded_pose, grounded = last_grounded_pose(poses, frame)
    ball_uv, ball_frame = nearest_ball(ball, frame)
    if pose is None or grounded_pose is None:
        return None

    foot_contact_px, foot_method = foot_pixel(pose)
    foot_ground_px, ground_foot_method = foot_pixel(grounded_pose)
    head_px, head_method = head_pixel(pose)
    body_px = float(foot_contact_px[1] - head_px[1])
    if body_px < 12:
        return None

    racket = racket_pixel(pose, ball_uv)
    racket_uv = racket[1] if racket else None
    # A WASB observation within two frames is primary. Farther observations are already
    # outgoing/incoming flight and are blended weakly with the pose-derived sweet spot.
    ball_delta = abs(ball_frame - frame) if ball_frame is not None else math.inf
    if ball_uv is not None and racket_uv is not None:
        ball_weight = 0.8 if ball_delta <= 2 else 0.3
        contact_uv = ball_weight * ball_uv + (1 - ball_weight) * racket_uv
        pixel_method = "ball_racket_blend"
    elif ball_uv is not None:
        contact_uv, pixel_method = ball_uv, "ball_only"
    elif racket_uv is not None:
        contact_uv, pixel_method = racket_uv, "racket_extrapolation"
    else:
        return None

    z_visual = HEAD_HEIGHT_FRACTION * height_m * (
        (foot_contact_px[1] - contact_uv[1]) / body_px
    )
    if contact["phase"] == "serve":
        # A conventional overhead serve is struck above the head. Missing toss/impact
        # pixels otherwise pull the blend down toward a post-contact wrist observation.
        z = float(np.clip(z_visual, 1.10 * height_m, 1.70 * height_m))
    else:
        z = float(np.clip(z_visual, 0.15, 1.35 * height_m))
    point = ray_at_z(camera.p_at(frame), contact_uv, z)
    foot_xy = cv2.perspectiveTransform(
        np.float32([[foot_ground_px]]), camera.h_at(grounded_pose.frame)
    )[0, 0].astype(float)
    if point is None or not np.all(np.isfinite(point)):
        point = np.r_[foot_xy, z]

    delta = point[:2] - foot_xy
    raw_reach = float(np.linalg.norm(delta))
    side = contact["side"]
    # Court-forward is +y for the near player and -y for the far player. A racket may
    # trail the grounded foot slightly, but a 2 m backward contact is a camera-ray ghost.
    direction = 1.0 if side == "near" else -1.0
    lateral_max = 1.55 if side == "near" else 2.05
    forward_max = 1.80 if side == "near" else 2.15
    backward_max = 0.45 if side == "near" else 0.65
    lateral = float(np.clip(delta[0], -lateral_max, lateral_max))
    forward = float(np.clip(direction * delta[1], -backward_max, forward_max))
    # The ray supplies direction, while a neutral strike prior prevents every bound-riding
    # monocular solution becoming the volume center. Far visual offsets are deliberately
    # shrunk more because a few pixels correspond to much larger court displacement.
    lateral_visual_weight = 0.75 if side == "near" else 0.50
    forward_visual_weight = 0.60 if side == "near" else 0.35
    forward_prior = 0.30 if contact["phase"] == "serve" else 0.55
    lateral *= lateral_visual_weight
    forward = forward_visual_weight * forward + (1 - forward_visual_weight) * forward_prior
    point[:2] = foot_xy + np.array([lateral, direction * forward])
    reach_max = math.hypot(lateral_max, forward_max)

    pose_conf = min(pose.conf, grounded_pose.conf)
    wrist_conf = racket[3] if racket else 0.0
    timing_inflation = 1.0 + 0.15 * min(ball_delta, 6) if math.isfinite(ball_delta) else 2.0
    ground_inflation = 1.0 if grounded else 1.5
    pose_inflation = 1.0 + max(0.0, 0.7 - pose_conf)
    if side == "near":
        sigma = np.array([0.35, 0.45, 0.28])
    else:
        sigma = np.array([0.65, 0.90, 0.48])
    sigma *= timing_inflation * ground_inflation * pose_inflation
    if racket is None or wrist_conf < 0.4:
        sigma *= 1.25
    if raw_reach > reach_max:
        sigma[:2] *= 1.25

    projected = rich.project_one(camera.p_at(frame), point)
    return {
        "center": point, "sigma": sigma, "contact_uv": contact_uv,
        "projected_uv": projected, "ball_uv": ball_uv, "ball_frame": ball_frame,
        "ball_delta": ball_delta, "racket_uv": racket_uv,
        "active_arm": racket[4] if racket else "", "wrist_conf": wrist_conf,
        "pose_frame": pose.frame, "grounded_frame": grounded_pose.frame,
        "grounded": grounded, "foot_xy": foot_xy, "foot_px": foot_ground_px,
        "foot_method": f"{ground_foot_method}/{foot_method}", "head_px": head_px,
        "head_method": head_method, "body_px": body_px, "z_visual": z_visual,
        "raw_reach": raw_reach, "reach_max": reach_max, "pixel_method": pixel_method,
        "pose_conf": pose_conf,
    }


def pose_paths(spec: rich.MatchSpec):
    if spec.out.endswith("rg2025f"):
        return (
            os.path.join(spec.out, "player_pose_50_fixed20_yolo26m_1280_on_court.csv"),
            os.path.join(spec.out, "player_pose_far_native_v1.csv"),
        )
    return (
        os.path.join(spec.out, "player_pose_60_serve1079_yolov8m_1280_on_court.csv"),
        os.path.join(spec.out, "player_pose_far_native_v1.csv"),
    )


def write_audit(spec, clip, rows, camera, ball, count=12,
                tag="contact_geometry_anthro_v1"):
    supported = [row for row in rows if row.get("center") is not None]
    if not supported:
        return None
    chosen = np.linspace(0, len(supported) - 1, min(count, len(supported))).round().astype(int)
    frames_root, image_size = rich.highest_frames(spec)
    artifact_size = res.CANONICAL_SIZE
    strips = []
    for chosen_index in chosen:
        row = supported[int(chosen_index)]
        center_frame = int(round(row["frame"]))
        panels = []
        for frame in (center_frame - 1, center_frame, center_frame + 1):
            image = cv2.imread(os.path.join(frames_root, clip, f"f_{frame:04d}.jpg"))
            if image is None:
                image = np.zeros((image_size.height, image_size.width, 3), np.uint8)
            if frame in ball:
                ball_uv = res.scale_points(ball[frame], artifact_size, image_size)
                radius = int(round(res.pixel_length(7, artifact_size, image_size)))
                cv2.circle(image, tuple(np.round(ball_uv).astype(int)), radius,
                           (0, 210, 255), 2, cv2.LINE_AA)
            projected = res.scale_points(
                rich.project_one(camera.p_at(frame), row["center"]), artifact_size, image_size
            )
            marker_size = int(round(res.pixel_length(18, artifact_size, image_size)))
            cv2.drawMarker(image, tuple(np.round(projected).astype(int)), (30, 30, 240),
                           cv2.MARKER_CROSS, marker_size, 2)
            if frame == center_frame:
                contact_uv = res.scale_points(row["contact_uv"], artifact_size, image_size)
                radius = int(round(res.pixel_length(10, artifact_size, image_size)))
                cv2.circle(image, tuple(np.round(contact_uv).astype(int)), radius,
                           (255, 80, 30), 2, cv2.LINE_AA)
                cv2.putText(
                    image,
                    f"c{row['contact_index']} {row['side']} z={row['center'][2]:.2f} "
                    f"sigz={row['sigma'][2]:.2f} reach={row['raw_reach']:.2f}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                    cv2.LINE_AA,
                )
            panels.append(cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA))
        strips.append(np.hstack(panels))
    blank = np.zeros_like(strips[0])
    while len(strips) % 2:
        strips.append(blank)
    montage = np.vstack([np.hstack(strips[i:i + 2]) for i in range(0, len(strips), 2)])
    path = os.path.join(spec.out, f"{tag}_{clip}_audit{len(chosen)}.jpg")
    cv2.imwrite(path, montage)
    res.write_coordinate_manifest(
        f"{path}.coordinates.json", image_size=image_size, artifact_size=artifact_size,
        source=frames_root, extra={"clip": clip, "audit": os.path.basename(path)},
        subnative_flagged=True,
        subnative_justification=(
            "legacy contact audit overlay; Resolution Contract migration inventory item 16"
        ),
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", required=True, choices=["rg2025f", "uso2025f"])
    parser.add_argument("--clips", nargs="*")
    parser.add_argument("--output", default="contact_geometry_anthro_v1.csv")
    parser.add_argument(
        "--camera-file", default="camera_P_per_frame_v1.npz",
        help="per-frame projection NPZ within the processed match directory",
    )
    parser.add_argument("--audit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    spec = rich.match_spec(args.match)
    stage = StageRun(spec.out, "contact_geometry_anthro_v1", args, seed=args.seed)
    clips = args.clips
    if not clips:
        pdata = np.load(os.path.join(spec.out, args.camera_file))
        clips = sorted(set(pdata["clips"].astype(str).tolist()))
    near_path, far_path = pose_paths(spec)
    near_cluster, cluster_names = load_identities(spec.out)
    output_rows = []
    audits = []
    summary = []
    for clip in clips:
        camera = rich.FrameCamera(spec.out, clip, args.camera_file)
        ball = rich.load_ball(spec.ball, clip)
        contacts = rich.load_contacts(spec.features, clip)
        rich.retime_contacts(contacts, ball)
        near_poses = load_pose_file(near_path, clip, 1.0) if os.path.exists(near_path) else {}
        far_poses = load_pose_file(far_path, clip, 0.5) if os.path.exists(far_path) else {}
        rows = []
        for index, contact in enumerate(contacts):
            player = player_for_side(clip, contact["side"], near_cluster, cluster_names)
            height = PLAYER_HEIGHT_M.get(player, DEFAULT_HEIGHT_M)
            poses = near_poses if contact["side"] == "near" else far_poses
            volume = contact_volume(contact, poses, ball, camera, height)
            row = {
                "clip": clip, "contact_index": index, "frame": contact["frame"],
                "frame_detector": contact["frame_detector"], "side": contact["side"],
                "phase": contact["phase"], "player": player, "player_height_m": height,
                "center": None,
            }
            if volume is not None:
                row.update(volume)
            rows.append(row)
        output_rows.extend(rows)
        supported = [row for row in rows if row["center"] is not None]
        audits.append(write_audit(
            spec, clip, rows, camera, ball, args.audit,
            tag=os.path.splitext(args.output)[0],
        ))
        summary.append({
            "clip": clip, "contacts": len(rows), "supported": len(supported),
            "near_supported": sum(row["side"] == "near" for row in supported),
            "far_supported": sum(row["side"] == "far" for row in supported),
            "grounded": sum(row.get("grounded", False) for row in supported),
            "median_sigma_z_m": float(np.median([row["sigma"][2] for row in supported]))
            if supported else math.nan,
            "median_raw_reach_m": float(np.median([row["raw_reach"] for row in supported]))
            if supported else math.nan,
        })

    fields = [
        "clip", "contact_index", "frame", "frame_detector", "side", "phase", "player",
        "player_height_m", "x", "y", "z", "sigma_x", "sigma_y", "sigma_z",
        "contact_u", "contact_v", "projected_u", "projected_v", "ball_u", "ball_v",
        "ball_frame", "ball_delta", "racket_u", "racket_v", "active_arm", "wrist_conf",
        "pose_frame", "grounded_frame", "grounded", "foot_x", "foot_y", "foot_u",
        "foot_v", "foot_method", "head_u", "head_v", "head_method", "body_px",
        "z_visual", "raw_reach", "reach_max", "pixel_method", "pose_conf",
    ]
    output = os.path.join(spec.out, args.output)
    with open(output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in output_rows:
            if row["center"] is None:
                writer.writerow({key: row.get(key, "") for key in fields})
                continue
            center, sigma = row["center"], row["sigma"]
            uv, projected = row["contact_uv"], row["projected_uv"]
            ball_uv, racket_uv = row["ball_uv"], row["racket_uv"]
            flat = {key: row.get(key, "") for key in fields}
            flat.update(
                x=center[0], y=center[1], z=center[2],
                sigma_x=sigma[0], sigma_y=sigma[1], sigma_z=sigma[2],
                contact_u=uv[0], contact_v=uv[1], projected_u=projected[0],
                projected_v=projected[1],
                ball_u=ball_uv[0] if ball_uv is not None else "",
                ball_v=ball_uv[1] if ball_uv is not None else "",
                racket_u=racket_uv[0] if racket_uv is not None else "",
                racket_v=racket_uv[1] if racket_uv is not None else "",
                foot_x=row["foot_xy"][0], foot_y=row["foot_xy"][1],
                foot_u=row["foot_px"][0], foot_v=row["foot_px"][1],
                head_u=row["head_px"][0], head_v=row["head_px"][1],
            )
            writer.writerow(flat)
    print(f"{summary} -> {output}")
    stage.finish(outputs={"summary": summary, "output": output,
                          "audits": [path for path in audits if path]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
