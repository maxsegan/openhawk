"""Physics- and player-gated contact detection over fitted ball-flight segments.

Each fitted segment begins after a tracker gap, bounce, racket hit, or noise break. A start is
classified as:

- ``bounce`` when an adjacent incoming segment reaches the court plane and the velocity change
  is consistent with the impact model;
- ``hit`` when the segment begins near a player and, when an incoming segment exists, the
  velocity change is racket-impact feasible;
- ``unknown`` otherwise.

The script reports contact-count agreement on every aligned rally-window point, including
points with zero detected segments. With hand-checked ``clip,frame`` annotations it also reports
frame-tolerant contact precision/recall/F1 and a player-distance threshold frontier.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

import resolution as res

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "validation"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "physics"))

import impact  # noqa: E402
from ground_truth import ground_truth_stream  # noqa: E402
from run_manifest import StageRun  # noqa: E402

RMS_GATE = 6.0
SPEED_MIN = 4.0
Z_BOUNCE = 0.6
PLAYER_NEAR_PX = 110.0
JUNCTION_GAP_SECONDS = 0.5
MIN_HIT_GAP_SECONDS = 0.08
SEED = 0
COURT_X_MARGIN_M = 2.5
BASELINE_MARGIN_M = 8.0


def load_fits(out_dir: str, name: str = "traj_fits.csv") -> dict[str, list[dict]]:
    per_clip = defaultdict(list)
    with open(os.path.join(out_dir, name), newline="") as f:
        for row in csv.DictReader(f):
            per_clip[row["clip"]].append(row)
    return per_clip


def load_boxes(out_dir: str, pose_name: str, pose_fps: float, track_fps: float):
    from ball import load_pose_boxes

    raw = load_pose_boxes(out_dir, pose_name, pose_fps, track_fps)
    homography_path = os.path.join(out_dir, "court_H_per_point.npz")
    if not os.path.exists(homography_path):
        return raw
    data = np.load(homography_path)
    homographies = dict(zip(data["pts"].tolist(), data["H"]))
    filtered = {}
    for (clip, frame), candidates in raw.items():
        H = homographies.get(int(clip[2:]))
        if H is None:
            continue
        sides = defaultdict(list)
        for box in candidates:
            x0, y0, x1, y1 = box
            foot = np.float32([[[(x0 + x1) / 2, y1]]])
            cx, cy = cv2.perspectiveTransform(foot, H)[0, 0]
            # Players routinely stand 4-7m behind a baseline. The former 4m margin removed
            # the actual far player while retaining unrelated in-court people; 8m remains
            # below the RG ball-kid row at projected y=36-38m.
            if (
                -COURT_X_MARGIN_M <= cx <= 10.97 + COURT_X_MARGIN_M
                and -BASELINE_MARGIN_M <= cy <= 23.77 + BASELINE_MARGIN_M
            ):
                side = "near" if cy < 11.885 else "far"
                sides[side].append(box)
        # When ball kids overlap the expanded court margin, the players are the largest people
        # on their respective court halves.
        chosen = [
            max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
            for boxes in sides.values()
            if boxes
        ]
        if chosen:
            filtered[(clip, frame)] = chosen
    return filtered


def split_ground_truth(match_id: str) -> list[list]:
    stream, _ = ground_truth_stream(match_id)
    points, current = [], []
    for shot in stream:
        if shot.kind == "<point>":
            points.append(current)
            current = []
        else:
            current.append(shot)
    points.append(current)
    return points


def evaluation_points(
    out_dir: str, point_map: str, frames_dir: str, points_file: str = ""
) -> list[int]:
    if points_file:
        with open(points_file) as handle:
            return [
                int(line)
                for raw in handle
                if (line := raw.strip()) and not line.startswith("#")
            ]
    path = os.path.join(out_dir, point_map)
    if os.path.exists(path):
        with open(path, newline="") as f:
            return [
                int(row["pt"])
                for row in csv.DictReader(f)
                if row.get("rally_t_start") and row.get("rally_t_end")
            ]
    root = os.path.join(out_dir, frames_dir)
    return sorted(int(name[2:]) for name in os.listdir(root) if name.startswith("pt"))


def bounce_consistent(v_in: np.ndarray, v_out: np.ndarray) -> bool:
    """Whether a velocity transition is plausible for some ordinary topspin value."""
    vin_h = float(np.hypot(v_in[0], v_in[1]))
    vout_h = float(np.hypot(v_out[0], v_out[1]))
    if v_in[2] >= 0 or v_out[2] <= 0 or vin_h < 1e-3:
        return False
    for w1 in (-300.0, -100.0, 0.0, 100.0, 300.0, 500.0):
        try:
            bounce = impact.court_bounce(vin_h, -float(v_in[2]), w1, "hard")
        except ValueError:
            continue
        if abs(bounce.vx2 - vout_h) <= max(3.0, 0.35 * vin_h) and abs(
            bounce.vy2 - float(v_out[2])
        ) <= max(2.0, 0.4 * bounce.vy2):
            return True
    return False


def point_box_distance(x: float, y: float, box: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = box
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return float(math.hypot(dx, dy))


def player_distance(boxes, clip: str, frame: int, u: float, v: float) -> float:
    frame_name = f"f_{frame:04d}.jpg"
    candidates = boxes.get((clip, frame_name), [])
    if not candidates:
        return math.inf
    return min(point_box_distance(u, v, box) for box in candidates)


def _state(row: dict, prefix: str) -> np.ndarray:
    suffixes = ("x0", "y0", "z0") if prefix == "start" else ("xe", "ye", "ze")
    return np.array([float(row[name]) for name in suffixes])


def _velocity(row: dict, prefix: str) -> np.ndarray:
    suffixes = ("vx0", "vy0", "vz0") if prefix == "start" else ("vxe", "vye", "vze")
    return np.array([float(row[name]) for name in suffixes])


def find_predecessor(row: dict, segments: list[dict], fps: float) -> dict | None:
    f0 = int(row["f0"])
    start = _state(row, "start")
    candidates = []
    for previous in segments:
        fe = int(previous["fe"])
        gap = f0 - fe
        if gap < 1 or gap > round(JUNCTION_GAP_SECONDS * fps):
            continue
        distance = float(np.linalg.norm(start - _state(previous, "end")))
        max_distance = min(6.0, 1.0 + 40.0 * gap / fps)
        if distance > max_distance:
            continue
        same_track = previous["track_id"] == row["track_id"]
        candidates.append((not same_track, gap, distance, previous))
    return min(candidates, default=(None, None, None, None), key=lambda item: item[:3])[3]


def classify_segments(
    segments: list[dict],
    boxes,
    clip: str,
    fps: float,
    player_near_px: float = PLAYER_NEAR_PX,
    rms_gate: float = RMS_GATE,
) -> list[dict]:
    required_pixels = {"u0", "v0"}
    good = [
        row
        for row in segments
        if float(row["rms_px"]) <= rms_gate
        and float(row["speed0"]) >= SPEED_MIN
        and required_pixels <= row.keys()
    ]
    good.sort(key=lambda row: (int(row["f0"]), int(row["track_id"])))
    events = []
    for row in good:
        frame = int(row["f0"])
        u, v = float(row["u0"]), float(row["v0"])
        near = player_distance(boxes, clip, frame, u, v)
        previous = find_predecessor(row, good, fps)
        v_in = _velocity(previous, "end") if previous is not None else None
        v_out = _velocity(row, "start")
        kind = "unknown"
        if v_in is not None and float(row["z0"]) <= Z_BOUNCE and bounce_consistent(v_in, v_out):
            kind = "bounce"
        elif near <= player_near_px:
            feasible = v_in is None
            if v_in is not None:
                try:
                    feasible = impact.racket_hit_feasible(v_in, v_out)
                except ValueError:
                    feasible = False
            if feasible:
                kind = "hit"
        confidence = (1.0 / (1.0 + float(row["rms_px"]))) * (
            math.exp(-near / max(player_near_px, 1.0)) if math.isfinite(near) else 0.0
        )
        events.append(
            {
                "clip": clip,
                "frame": frame,
                "kind": kind,
                "track_id": int(row["track_id"]),
                "u": u,
                "v": v,
                "player_distance_px": near,
                "rms_px": float(row["rms_px"]),
                "confidence": confidence,
            }
        )

    hits = sorted((event for event in events if event["kind"] == "hit"), key=lambda e: e["frame"])
    deduped = []
    min_gap = max(1, round(MIN_HIT_GAP_SECONDS * fps))
    for event in hits:
        if deduped and event["frame"] - deduped[-1]["frame"] <= min_gap:
            if event["confidence"] > deduped[-1]["confidence"]:
                deduped[-1] = event
        else:
            deduped.append(event)
    return [event for event in events if event["kind"] != "hit"] + deduped


def rally_bucket(point: list) -> str:
    rally_shots = sum(shot.kind == "shot" for shot in point)
    if rally_shots <= 3:
        return "1-3"
    if rally_shots <= 8:
        return "4-8"
    return "9+"


def bootstrap_pm1(rows: list[dict], seed: int = SEED, samples: int = 2000) -> tuple[float, float]:
    values = np.array([row["within_pm1"] for row in rows], dtype=float)
    if not len(values):
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))


def count_report(
    events_by_clip: dict[str, list[dict]], eval_points: list[int], gt_points: list[list]
) -> tuple[list[dict], str]:
    rows = []
    for pt in eval_points:
        if pt < 1 or pt > len(gt_points):
            continue
        point = gt_points[pt - 1]
        expected = sum(shot.kind in {"serve", "shot"} for shot in point)
        if expected == 0:
            continue
        clip = f"pt{pt:04d}"
        predicted = sum(event["kind"] == "hit" for event in events_by_clip.get(clip, []))
        rows.append(
            {
                "pt": pt,
                "clip": clip,
                "predicted": predicted,
                "expected": expected,
                "difference": predicted - expected,
                "within_pm1": abs(predicted - expected) <= 1,
                "rally_bucket": rally_bucket(point),
            }
        )
    n = len(rows)
    within = sum(row["within_pm1"] for row in rows)
    lo, hi = bootstrap_pm1(rows)
    parts = [f"contact_count_pm1={within}/{n} ({within / max(n, 1):.1%}, 95% CI {lo:.1%}-{hi:.1%})"]
    for bucket in ("1-3", "4-8", "9+"):
        subset = [row for row in rows if row["rally_bucket"] == bucket]
        good = sum(row["within_pm1"] for row in subset)
        parts.append(f"{bucket}={good}/{len(subset)} ({good / max(len(subset), 1):.1%})")
    return rows, " | ".join(parts)


def load_contact_ground_truth(path: str) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        {
            **row,
            "frame": int(row["frame"]),
            "event_type": row.get("event_type", "") or "unknown",
            "side": row.get("side", "") or "unknown",
        }
        for row in rows
    ]


def contact_f1(
    predicted: list[dict],
    truth: list[dict],
    tolerance: int,
    evaluation_clips: set[str] | None = None,
) -> dict:
    pred_by_clip, truth_by_clip = defaultdict(list), defaultdict(list)
    for event in truth:
        truth_by_clip[event["clip"]].append(event)
    # A partially annotated file defines its evaluation scope by clip. Predictions from other
    # clips are not false positives; every contact in an included clip must still be annotated.
    annotated_clips = evaluation_clips if evaluation_clips is not None else set(truth_by_clip)
    for event in predicted:
        if event["kind"] == "hit" and event["clip"] in annotated_clips:
            pred_by_clip[event["clip"]].append(event)

    matches, used = [], set()
    for clip, targets in truth_by_clip.items():
        candidates = pred_by_clip.get(clip, [])
        for target in sorted(targets, key=lambda row: row["frame"]):
            options = [
                (abs(candidate["frame"] - target["frame"]), i, candidate)
                for i, candidate in enumerate(candidates)
                if (clip, i) not in used and abs(candidate["frame"] - target["frame"]) <= tolerance
            ]
            if options:
                _, i, candidate = min(options)
                used.add((clip, i))
                matches.append((candidate, target))
    n_pred = sum(len(events) for events in pred_by_clip.values())
    n_truth = len(truth)
    tp = len(matches)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_truth if n_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "n_pred": n_pred,
        "n_truth": n_truth,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
    }


def write_outputs(out_dir: str, events: list[dict], counts: list[dict]) -> None:
    event_path = os.path.join(out_dir, "contact_events_v2.csv")
    with open(event_path, "w", newline="") as f:
        fields = [
            "clip",
            "frame",
            "kind",
            "track_id",
            "u",
            "v",
            "player_distance_px",
            "rms_px",
            "confidence",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in events)
    count_path = os.path.join(out_dir, "contacts_v2.csv")
    with open(count_path, "w", newline="") as f:
        fields = ["pt", "clip", "predicted", "expected", "difference", "within_pm1", "rally_bucket"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(counts)


def write_event_audit(
    out_dir: str,
    frames_dir: str,
    events: list[dict],
    boxes,
    count: int,
    seed: int,
    tag: str = "",
    artifact_size: res.FrameSize = res.CANONICAL_SIZE,
) -> str | None:
    hits = [event for event in events if event["kind"] == "hit"]
    if not hits or count <= 0:
        return None
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(hits), size=min(count, len(hits)), replace=False)
    panels = []
    for index in selected:
        event = hits[int(index)]
        frame_name = f"f_{event['frame']:04d}.jpg"
        path = os.path.join(out_dir, frames_dir, event["clip"], frame_name)
        image = cv2.imread(path)
        if image is None:
            continue
        image_size = res.FrameSize(image.shape[1], image.shape[0])
        event_uv = res.scale_points((event["u"], event["v"]), artifact_size, image_size)
        radius = int(round(res.pixel_length(15, artifact_size, image_size)))
        cv2.circle(image, tuple(np.round(event_uv).astype(int)), radius, (0, 255, 255), 2)
        for x0, y0, x1, y1 in boxes.get((event["clip"], frame_name), []):
            box = res.scale_boxes((x0, y0, x1, y1), artifact_size, image_size)
            cv2.rectangle(image, tuple(np.round(box[:2]).astype(int)),
                          tuple(np.round(box[2:]).astype(int)), (255, 0, 255), 2)
        label = (
            f"{event['clip']}/{frame_name} hit d={event['player_distance_px']:.0f}px "
            f"rms={event['rms_px']:.1f}"
        )
        cv2.rectangle(image, (0, 0), (620, 28), (0, 0, 0), -1)
        cv2.putText(image, label, (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        panels.append(cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA))
    if not panels:
        return None
    blank = np.zeros_like(panels[0])
    while len(panels) % 3:
        panels.append(blank)
    montage = np.vstack([np.hstack(panels[i : i + 3]) for i in range(0, len(panels), 3)])
    name_tag = f"_{tag}" if tag else ""
    output = os.path.join(
        out_dir, f"contact_event_audit{name_tag}_n{len(selected)}_seed{seed}.jpg"
    )
    cv2.imwrite(output, montage)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--match", required=True)
    parser.add_argument("--fits", default="traj_fits.csv")
    parser.add_argument("--pose", default="player_pose_25.csv")
    parser.add_argument("--pose-fps", type=float, default=25.0)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--frames-dir", default="rally_frames_60")
    parser.add_argument("--point-map", default="point_video_map.csv")
    parser.add_argument("--points-file", default="", help="optional fixed evaluation point IDs")
    parser.add_argument("--player-near-px", type=float, default=PLAYER_NEAR_PX)
    parser.add_argument("--rms-gate", type=float, default=RMS_GATE)
    parser.add_argument("--contact-ground-truth", default="")
    parser.add_argument("--tolerance-frames", type=int, default=5)
    parser.add_argument("--sweep-player-near", default="60,80,100,110,130,160")
    parser.add_argument("--audit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--report-mcp-counts", action="store_true")
    args = parser.parse_args()
    stage_run = StageRun(args.out, "contacts_v2", args, seed=SEED)

    fits = load_fits(args.out, args.fits)
    boxes = load_boxes(args.out, args.pose, args.pose_fps, args.fps)
    gt_points = split_ground_truth(args.match) if args.report_mcp_counts else []
    eval_points = evaluation_points(args.out, args.point_map, args.frames_dir, args.points_file)

    truth = (
        load_contact_ground_truth(args.contact_ground_truth) if args.contact_ground_truth else []
    )
    thresholds = [float(value) for value in args.sweep_player_near.split(",") if value]
    if args.player_near_px not in thresholds:
        thresholds.append(args.player_near_px)

    selected_events, selected_counts, selected_summary = [], [], ""
    for threshold in sorted(set(thresholds)):
        events_by_clip = {
            clip: classify_segments(
                segments, boxes, clip, args.fps, threshold, rms_gate=args.rms_gate
            )
            for clip, segments in fits.items()
        }
        counts, count_summary = count_report(events_by_clip, eval_points, gt_points)
        flat_events = [event for events in events_by_clip.values() for event in events]
        line = f"player_near_px={threshold:g} | {count_summary}"
        if truth:
            frame_score = contact_f1(flat_events, truth, args.tolerance_frames)
            line += (
                f" | contact_f1 P={frame_score['precision']:.3f} "
                f"R={frame_score['recall']:.3f} F1={frame_score['f1']:.3f} "
                f"TP={frame_score['tp']} pred={frame_score['n_pred']} gt={frame_score['n_truth']}"
            )
        print(line)
        if threshold == args.player_near_px:
            selected_events = flat_events
            selected_counts = counts
            selected_summary = line

    write_outputs(args.out, selected_events, selected_counts)
    audit_path = write_event_audit(
        args.out, args.frames_dir, selected_events, boxes, args.audit, args.seed
    )
    if audit_path:
        print(f"audit -> {audit_path}")
    stage_run.finish(
        outputs={
            "evaluation_points": len(selected_counts),
            "events": len(selected_events),
            "summary": selected_summary,
            "audit_samples": min(args.audit, sum(e["kind"] == "hit" for e in selected_events)),
            "audit_path": audit_path,
            "contact_ground_truth_n": len(truth),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
