"""Source-only anchor extension over ownership-annotated tracker rows (default off).

After ``ball_ownership`` has decided which tracker segments are owned moving objects, every
end of every consumed moving segment is an *anchor end*.  This module walks outward from each
anchor end, forward from its tail and backward from its head, into frames that currently hold
no consumed row (empty, or withheld as stationary), selecting **observed** native candidates
whose camera-compensated motion continues the anchor.  A frame receives a row only when an
actual candidate row is accepted; no predicted position is ever written, no existing consumed
row is moved, replaced or re-scored.

Why bidirectional: an anchor on the far side of a contact recovers the frames between the
contact and itself, while the near side runs into the impulse and stops.  A contact or bounce
therefore shows up as an honest break between a forward chain and a backward chain, which is
what the flight fitter already treats as an anchor.

What it does not know: nothing here reads labels, fitted trajectories, truth contacts or the
coarse lock.  Ownership cannot say which moving chain is the ball, so an extension inherits the
chain id of its anchor and nothing more.  A moving-clutter anchor can therefore be extended;
the defence is that every accepted row must be an observed candidate on the anchor's own
camera-compensated line, the chain must survive against rival hypotheses, and the decision
table records every abstention.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from cv.pipeline import ball_motion_tracker as tracker
from cv.pipeline import ball_ownership as bo
from cv.pipeline.track_artifact import write_track_artifact

SUPPORT_TOKEN = "support:anchor_extension"
ANCHOR_TOKEN = "anchor_segment"
REGIME = "anchor_extension"
SIDE_COLUMNS = (
    "clip",
    "frame",
    "anchor_run",
    "anchor_chain_id",
    "direction",
    "step",
    "x_native",
    "y_native",
    "innovation_px",
    "gate_px",
    "rivals_in_gate",
    "detector_support",
    "sources",
    "hypothesis_cost",
)
DECISION_COLUMNS = (
    "clip",
    "anchor_run",
    "anchor_chain_id",
    "direction",
    "anchor_frame",
    "anchor_speed_native",
    "verdict",
    "reason",
    "rows_added",
    "rows_considered",
    "best_cost",
    "rival_cost",
    "stop_reason",
    "best_path",
)


@dataclass(frozen=True)
class ExtensionConfig:
    base_gate_native: float = 12.0
    gate_velocity_fraction: float = 0.25
    miss_gate_growth: float = 1.5
    max_consecutive_misses: int = 2
    beam: int = 3
    min_rows: int = 3
    kink_angle_deg: float = 60.0
    kink_speed_ratio: float = 2.0
    rival_margin: float = 1.0
    miss_penalty: float = 1.0
    minimum_detectors: int = 1
    max_steps: int = 48


@dataclass
class Hypothesis:
    history: list[tuple[int, np.ndarray]]  # (frame, native xy) including the anchor rows
    accepted: list[dict] = field(default_factory=list)
    misses: int = 0
    cost: float = 0.0
    alive: bool = True
    stop_reason: str = ""

    def differs_from(self, other: "Hypothesis", radius: float) -> bool:
        mine = {row["frame"]: row["xy"] for row in self.accepted}
        for row in other.accepted:
            xy = mine.get(row["frame"])
            if xy is not None and float(np.linalg.norm(xy - row["xy"])) > radius:
                return True
        return False


@dataclass(frozen=True)
class AnchorEnd:
    run_index: int
    chain_id: int
    direction: int  # +1 extends forward from the tail, -1 backward from the head
    history: list[tuple[int, np.ndarray]]
    speed_native: float


def _observation_pool(
    frame_observations: dict[int, list[tracker.Observation]], config: tracker.MotionConfig
) -> dict[int, list[tracker.Observation]]:
    """The candidate set the tracker itself saw, minus the guide.

    Merging by the tracker's cluster radius keeps coincident streams of one model as one
    observation, so ``detector_count`` counts distinct models, never distinct arms.
    """
    merged = {
        # Drop the guide before clustering. Otherwise its flag contaminates a
        # coincident independent detector observation and deletes real evidence.
        frame: tracker.merge_observations(
            [item for item in observations if not item.is_guide], config.cluster_radius_native
        )
        for frame, observations in frame_observations.items()
    }
    merged = tracker.suppress_static_hotspots(merged, config)
    return {
        frame: [item for item in observations if not item.is_guide]
        for frame, observations in merged.items()
    }


def consumed_runs(ownership_rows: list[dict]) -> list[list[dict]]:
    """Maximal runs of consecutive-frame consumed rows sharing one chain id."""
    runs: list[list[dict]] = []
    previous = None
    for row in ownership_rows:
        if row["owner"] in bo.WITHHELD_OWNERS:
            continue
        frame = bo._frame(row)
        if (
            runs
            and previous is not None
            and frame == previous[0] + 1
            and row["chain_id"] == previous[1]
        ):
            runs[-1].append(row)
        else:
            runs.append([row])
        previous = (frame, row["chain_id"])
    return runs


def anchor_ends(
    ownership_rows: list[dict],
    geometry: tracker.Geometry,
    clip: str,
    pan: bo.PanModel,
    config: tracker.MotionConfig,
) -> list[AnchorEnd]:
    """Both ends of every consumed run whose local wing moves under camera compensation."""
    count = max(2, config.reentry_support_frames)
    ends: list[AnchorEnd] = []
    for index, run in enumerate(consumed_runs(ownership_rows)):
        if run[0]["chain_motion"] != "moving" or len(run) < 2:
            continue
        for direction in (+1, -1):
            wing = run[-count:] if direction > 0 else run[:count]
            if bo.motion_state(wing, geometry, clip, pan, config).state != "moving":
                continue
            ordered = wing[::-1] if direction < 0 else wing
            history = [(bo._frame(row), bo._native_xy(row)) for row in ordered[-2:]]
            last_frame, last_xy = history[-1]
            prev_frame, prev_xy = history[-2]
            prev_in_last = bo.transport(geometry, clip, prev_frame, last_frame, prev_xy)
            if prev_in_last is None:
                prev_in_last = prev_xy
            speed = float(np.linalg.norm(last_xy - prev_in_last)) / abs(last_frame - prev_frame)
            ends.append(AnchorEnd(index, int(run[0]["chain_id"]), direction, history, speed))
    return ends


def _predict(
    history: list[tuple[int, np.ndarray]],
    frame: int,
    geometry: tracker.Geometry,
    clip: str,
    pan: bo.PanModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Constant-velocity prediction in ``frame``'s camera from the last two accepted rows.

    Returns (prediction, velocity per frame, last position in this camera, pan allowance);
    the allowance is zero when both rows transport under a reliable camera.
    """
    (f0, p0), (f1, p1) = history[-2], history[-1]
    p1_here = bo.transport(geometry, clip, f1, frame, p1)
    p0_here = bo.transport(geometry, clip, f0, frame, p0)
    allowance = 0.0
    if p1_here is None or p0_here is None:
        p1_here, p0_here = p1, p0
        allowance = pan.allowance(clip, f1, frame)
    velocity = (p1_here - p0_here) / float(f1 - f0)
    prediction = p1_here + velocity * float(frame - f1)
    return prediction, velocity, p1_here, allowance


def _is_kink(
    velocity: np.ndarray,
    last_here: np.ndarray,
    candidate: np.ndarray,
    dt: float,
    cfg: ExtensionConfig,
) -> bool:
    new_velocity = (candidate - last_here) / dt
    speed, new_speed = float(np.linalg.norm(velocity)), float(np.linalg.norm(new_velocity))
    if speed < 2.0 and new_speed < 2.0:
        return False
    # Near-zero displacement makes direction unreliable, not a sudden speed
    # collapse harmless. Use the existing noise floor for the ratio as well.
    ratio = max(new_speed, 2.0) / max(speed, 2.0)
    if ratio > cfg.kink_speed_ratio or ratio < 1.0 / cfg.kink_speed_ratio:
        return True
    if speed < 2.0 or new_speed < 2.0:
        return False
    cosine = float(np.dot(velocity, new_velocity)) / (speed * new_speed)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine)))) > cfg.kink_angle_deg


def _walk(
    end: AnchorEnd,
    pool: dict[int, list[tracker.Observation]],
    occupied: set[int],
    frame_range: tuple[int, int],
    geometry: tracker.Geometry,
    clip: str,
    pan: bo.PanModel,
    config: tracker.MotionConfig,
    cfg: ExtensionConfig,
) -> tuple[list[Hypothesis], str]:
    beam = [Hypothesis(history=list(end.history))]
    finished: list[Hypothesis] = []
    frame = end.history[-1][0]
    stop_reason = "max_steps"
    for _ in range(cfg.max_steps):
        frame += end.direction
        if frame < frame_range[0] or frame > frame_range[1]:
            stop_reason = "clip_end"
            break
        if frame in occupied:
            stop_reason = "occupied"
            break
        children: dict[tuple, Hypothesis] = {}
        for hyp in beam:
            prediction, velocity, last_here, allowance = _predict(
                hyp.history, frame, geometry, clip, pan
            )
            if not math.isfinite(allowance):
                # Unbracketed unreliable camera: no measurable pan, so no finite gate and
                # no honest innovation. The hypothesis ends here rather than accepting anything.
                hyp.alive, hyp.stop_reason = False, "camera_unmeasured"
                finished.append(hyp)
                continue
            if np.any(tracker._image_edge_distances(prediction) < -cfg.base_gate_native):
                hyp.alive, hyp.stop_reason = False, "image_exit"
                finished.append(hyp)
                continue
            gate = (
                cfg.base_gate_native + cfg.gate_velocity_fraction * float(np.linalg.norm(velocity))
            ) * (cfg.miss_gate_growth**hyp.misses) + allowance
            dt = float(frame - hyp.history[-1][0])
            in_gate = []
            kinked = 0
            for item in pool.get(frame, []):
                if item.detector_count < cfg.minimum_detectors:
                    continue
                xy = np.asarray([item.x, item.y])
                distance = float(np.linalg.norm(xy - prediction))
                if distance > gate:
                    continue
                if _is_kink(velocity, last_here, xy, dt, cfg):
                    kinked += 1
                    continue
                in_gate.append((distance, item, xy))
            in_gate.sort(key=lambda entry: entry[0])
            for distance, item, xy in in_gate[: cfg.beam]:
                key = (frame, round(float(xy[0]), 3), round(float(xy[1]), 3))
                child = Hypothesis(
                    history=[*hyp.history, (frame, xy)],
                    accepted=[
                        *hyp.accepted,
                        {
                            "frame": frame,
                            "observation": item,
                            "xy": xy,
                            "innovation_px": distance,
                            "gate_px": gate,
                            "rivals_in_gate": len(in_gate) - 1,
                        },
                    ],
                    misses=0,
                    cost=hyp.cost + distance / gate,
                )
                if key not in children or children[key].cost > child.cost:
                    children[key] = child
            if not in_gate:
                if hyp.misses + 1 > cfg.max_consecutive_misses:
                    hyp.alive = False
                    hyp.stop_reason = "kink" if kinked else "misses"
                    finished.append(hyp)
                else:
                    key = ("miss", id(hyp))
                    children[key] = Hypothesis(
                        history=hyp.history,
                        accepted=hyp.accepted,
                        misses=hyp.misses + 1,
                        cost=hyp.cost + cfg.miss_penalty,
                    )
        beam = sorted(children.values(), key=lambda item: item.cost)[: cfg.beam]
        if not beam:
            reasons = [hyp.stop_reason for hyp in finished if hyp.stop_reason]
            stop_reason = max(set(reasons), key=reasons.count) if reasons else "exhausted"
            break
    for hyp in beam:
        hyp.alive = False
        hyp.stop_reason = stop_reason
        finished.append(hyp)
    return finished, stop_reason


def _decide(
    finished: list[Hypothesis], config: tracker.MotionConfig, cfg: ExtensionConfig
) -> tuple[Hypothesis | None, str, float]:
    """Best hypothesis, or None with the abstention reason; also the nearest rival's cost."""
    eligible = [hyp for hyp in finished if len(hyp.accepted) >= cfg.min_rows]
    if not eligible:
        longest = max((len(hyp.accepted) for hyp in finished), default=0)
        return None, f"too_short:{longest}", math.nan
    eligible.sort(key=lambda hyp: (-len(hyp.accepted), hyp.cost))
    best = eligible[0]
    rival_cost = math.nan
    for rival in eligible[1:]:
        if not rival.differs_from(best, config.cluster_radius_native):
            continue
        rival_cost = rival.cost if math.isnan(rival_cost) else min(rival_cost, rival.cost)
        if rival.cost - best.cost < cfg.rival_margin:
            best.stop_reason = f"{best.stop_reason}|rival={_path(rival)}"
            return None, "rival_tie", rival_cost
    return best, "accepted", rival_cost


def _path(hyp: Hypothesis) -> str:
    return " ".join(f"{row['frame']}:{row['xy'][0]:.0f},{row['xy'][1]:.0f}" for row in hyp.accepted)


def _extension_rows_moving(
    best: Hypothesis, geometry: tracker.Geometry, clip: str, pan: bo.PanModel, config
) -> bool:
    rows = [
        {"frame": f"f_{row['frame']:04d}.jpg", "x": row["xy"][0] / 2.0, "y": row["xy"][1] / 2.0}
        for row in sorted(best.accepted, key=lambda row: row["frame"])
    ]
    # A finite pan allowance permits candidate search, but ambiguous motion is not
    # evidence that the continuation is the ball. Apply the same rule as anchors.
    return bo.motion_state(rows, geometry, clip, pan, config).state == "moving"


def _public_row(
    template: list[str],
    clip: str,
    end: AnchorEnd,
    accepted: dict,
    geometry: tracker.Geometry,
) -> dict:
    item: tracker.Observation = accepted["observation"]
    gate = accepted["gate_px"]
    direction = "fwd" if end.direction > 0 else "bwd"
    sources = [
        *item.sources,
        f"{SUPPORT_TOKEN}_{direction}",
        f"{ANCHOR_TOKEN}:{end.run_index}",
        *(f"crop_region:{value}" for value in item.crop_provenance),
    ]
    values = {
        "clip": clip,
        "frame": f"f_{accepted['frame']:04d}.jpg",
        "x": item.x / 2.0,
        "y": item.y / 2.0,
        "track_id": end.chain_id,
        "score": item.score,
        "rank": item.rank,
        "sources": "+".join(sources),
        "regime": REGIME,
        "regime_probability": 0.0,
        "innovation_mahalanobis": accepted["innovation_px"] / gate,
        "innovation_cov_xx_native": gate * gate,
        "innovation_cov_xy_native": 0.0,
        "innovation_cov_yy_native": gate * gate,
        "filter_cov_trace_native": 2.0 * gate * gate,
        "confidence": 1.0 / math.sqrt(2.0 * gate * gate),
        "detector_support": item.detector_count,
        "homography_source": geometry.source,
    }
    return {key: values.get(key, "") for key in template}


def extend_clip(
    ownership_rows: list[dict],
    frame_observations: dict[int, list[tracker.Observation]],
    geometry: tracker.Geometry,
    clip: str,
    config: tracker.MotionConfig,
    pan: bo.PanModel | None = None,
    cfg: ExtensionConfig = ExtensionConfig(),
) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (extended consumed rows, side rows, anchor decisions) for one clip.

    The extended rows are the clip's consumed rows plus accepted extension rows, in frame
    order and with the consumed rows' exact columns.  Consumed rows are returned unchanged.
    """
    if pan is None:
        pan = bo.pan_model(geometry)
    consumed = bo.consumed_rows(ownership_rows)
    if not consumed or not frame_observations:
        return consumed, [], []
    template = list(consumed[0].keys())
    pool = _observation_pool(frame_observations, config)
    occupied = {bo._frame(row) for row in consumed}
    frame_range = (min(pool), max(pool))
    claims: dict[int, list[tuple[AnchorEnd, dict]]] = {}
    decisions: list[dict] = []
    ends = anchor_ends(ownership_rows, geometry, clip, pan, config)
    for end in ends:
        finished, stop_reason = _walk(
            end, pool, occupied, frame_range, geometry, clip, pan, config, cfg
        )
        best, reason, rival_cost = _decide(finished, config, cfg)
        longest = max(finished, key=lambda hyp: (len(hyp.accepted), -hyp.cost), default=None)
        best_path = (
            ""
            if longest is None
            else _path(longest)
            + (f" |{longest.stop_reason.partition('|')[2]}" if "|" in longest.stop_reason else "")
        )
        if best is not None and not _extension_rows_moving(best, geometry, clip, pan, config):
            best, reason = None, "extension_motion_unconfirmed"
        decisions.append(
            {
                "clip": clip,
                "anchor_run": end.run_index,
                "anchor_chain_id": end.chain_id,
                "direction": "fwd" if end.direction > 0 else "bwd",
                "anchor_frame": end.history[-1][0],
                "anchor_speed_native": f"{end.speed_native:.2f}",
                "verdict": "accepted" if best is not None else "abstain",
                "reason": reason,
                "rows_added": len(best.accepted) if best is not None else 0,
                "rows_considered": max((len(hyp.accepted) for hyp in finished), default=0),
                "best_cost": "" if best is None else f"{best.cost:.3f}",
                "rival_cost": "" if math.isnan(rival_cost) else f"{rival_cost:.3f}",
                "stop_reason": stop_reason,
                "best_path": best_path,
            }
        )
        if best is None:
            continue
        for accepted in best.accepted:
            claims.setdefault(accepted["frame"], []).append((end, accepted))
    # Two anchors reaching the same frame must agree on the observation; otherwise that
    # frame stays empty. An extension that loses rows below the minimum is dropped whole.
    resolved: dict[int, tuple[AnchorEnd, dict]] = {}
    for frame, entries in claims.items():
        first_xy = entries[0][1]["xy"]
        if all(
            float(np.linalg.norm(entry[1]["xy"] - first_xy)) <= config.cluster_radius_native
            for entry in entries[1:]
        ):
            resolved[frame] = min(entries, key=lambda entry: entry[1]["innovation_px"])
    per_end: dict[tuple[int, int], int] = {}
    for end, _ in resolved.values():
        per_end[(end.run_index, end.direction)] = per_end.get((end.run_index, end.direction), 0) + 1
    side_rows: list[dict] = []
    new_rows: list[dict] = []
    for frame in sorted(resolved):
        end, accepted = resolved[frame]
        if per_end[(end.run_index, end.direction)] < cfg.min_rows:
            continue
        new_rows.append(_public_row(template, clip, end, accepted, geometry))
        item = accepted["observation"]
        side_rows.append(
            {
                "clip": clip,
                "frame": f"f_{frame:04d}.jpg",
                "anchor_run": end.run_index,
                "anchor_chain_id": end.chain_id,
                "direction": "fwd" if end.direction > 0 else "bwd",
                "step": abs(frame - end.history[-1][0]),
                "x_native": f"{item.x:.3f}",
                "y_native": f"{item.y:.3f}",
                "innovation_px": f"{accepted['innovation_px']:.2f}",
                "gate_px": f"{accepted['gate_px']:.2f}",
                "rivals_in_gate": accepted["rivals_in_gate"],
                "detector_support": item.detector_count,
                "sources": "+".join(item.sources),
                "hypothesis_cost": "",
            }
        )
    for decision in decisions:
        key = (decision["anchor_run"], +1 if decision["direction"] == "fwd" else -1)
        if decision["verdict"] == "accepted" and per_end.get(key, 0) < cfg.min_rows:
            decision["verdict"], decision["reason"] = "abstain", "conflict_between_anchors"
            decision["rows_added"] = 0
        elif decision["verdict"] == "accepted":
            decision["rows_added"] = per_end[key]
    extended = sorted([*consumed, *new_rows], key=lambda row: bo._frame(row))
    return extended, side_rows, decisions


def read_rows(path: Path, clips: set[str] | None = None) -> dict[str, list[dict]]:
    """Read a track artifact into legacy-column rows (native mirror columns dropped)."""
    by_clip: dict[str, list[dict]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if clips is not None and row["clip"] not in clips:
                continue
            row.pop("x_native", None)
            row.pop("y_native", None)
            by_clip.setdefault(row["clip"], []).append(row)
    return by_clip


def run_extension(
    track_path: Path,
    ownership_path: Path,
    candidate_paths: list[Path],
    source_names: list[str],
    geometry: tracker.Geometry,
    output: Path,
    side_output: Path,
    config: tracker.MotionConfig,
    cfg: ExtensionConfig = ExtensionConfig(),
    clips: set[str] | None = None,
) -> dict:
    ownership = read_rows(ownership_path, clips)
    consumed = read_rows(track_path, clips)
    streams = tracker.load_observations(candidate_paths, source_names, clips)
    pan = bo.pan_model(geometry)
    all_rows: list[dict] = []
    side_rows: list[dict] = []
    decisions: list[dict] = []
    per_clip = {}
    for clip in sorted(set(ownership) | set(consumed)):
        clip_ownership = ownership.get(clip, [])
        before = consumed.get(clip, [])
        extended, clip_side, clip_decisions = extend_clip(
            clip_ownership, streams.get(clip, {}), geometry, clip, config, pan, cfg
        )
        if len(extended) < len(before):
            raise RuntimeError(f"{clip}: extension dropped consumed rows")
        all_rows.extend(extended)
        side_rows.extend(clip_side)
        decisions.extend(clip_decisions)
        per_clip[clip] = {
            "consumed_before": len(before),
            "rows_after": len(extended),
            "rows_added": len(extended) - len(before),
            "anchor_ends": len(clip_decisions),
            "accepted_ends": sum(1 for d in clip_decisions if d["verdict"] == "accepted"),
        }
    write_track_artifact(output, all_rows, candidate_paths)
    side_output.parent.mkdir(parents=True, exist_ok=True)
    with side_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SIDE_COLUMNS))
        writer.writeheader()
        writer.writerows(side_rows)
    decisions_path = side_output.with_name(side_output.stem + "_decisions.csv")
    with decisions_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DECISION_COLUMNS))
        writer.writeheader()
        writer.writerows(decisions)
    return {
        "schema": "ball_anchor_extension_v1",
        "labels_loaded": False,
        "predicted_rows_emitted": 0,
        "config": cfg.__dict__,
        "candidate_arms": [
            {"path": str(path), "source": source}
            for path, source in zip(candidate_paths, source_names, strict=True)
        ],
        "rows_before": sum(len(rows) for rows in consumed.values()),
        "rows_after": len(all_rows),
        "rows_added": len(side_rows),
        "decisions": {
            "accepted": sum(1 for d in decisions if d["verdict"] == "accepted"),
            "abstain": sum(1 for d in decisions if d["verdict"] == "abstain"),
        },
        "clips": per_clip,
        "side_artifact": side_output.name,
        "decisions_artifact": decisions_path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", type=Path, required=True, help="consumed track (ownership on)")
    parser.add_argument("--ownership", type=Path, required=True, help="ownership side artifact")
    parser.add_argument("--candidates", type=Path, action="append", required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--extra-candidates", type=Path, action="append", default=[])
    parser.add_argument("--extra-source", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--side-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--court-homographies", type=Path)
    parser.add_argument("--camera-projections", type=Path)
    parser.add_argument("--clip", action="append", default=[])
    parser.add_argument("--crop-first", action="store_true")
    parser.add_argument("--lock-is-crop", action="store_true")
    args = parser.parse_args()
    if len(args.candidates) != len(args.source) or len(args.extra_candidates) != len(
        args.extra_source
    ):
        parser.error("every --candidates needs one --source (same for --extra-*)")
    geometry = tracker.load_geometry(args.court_homographies, args.camera_projections)
    config = (
        tracker.crop_first_config(lock_is_crop=args.lock_is_crop)
        if args.crop_first
        else tracker.MotionConfig()
    )
    config = replace(config, primary_ball_ownership=True)
    report = run_extension(
        args.track,
        args.ownership,
        [*args.candidates, *args.extra_candidates],
        [*args.source, *args.extra_source],
        geometry,
        args.output,
        args.side_output,
        config,
        clips=set(args.clip) or None,
    )
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
