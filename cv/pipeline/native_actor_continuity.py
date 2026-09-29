"""Conservative image-space actor witnesses across short metric-camera abstentions.

Observed boxes only: reliable sided anchors and unique forward/backward native
association must agree. This does not produce court coordinates or upgrade H.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cv.pipeline.player_tracker import iou
from cv.pipeline.resolution import coordinate_manifest_path

MAX_GAP_SECONDS = 1.0
MIN_IOU = 0.3
MIN_IOU_MARGIN = 0.15
MIN_CONFIDENCE = 0.1
NATIVE_COLUMNS = ("x0_native", "y0_native", "x1_native", "y1_native")


@dataclass(frozen=True)
class NativeBox:
    box: tuple[float, float, float, float]
    confidence: float
    track_id: str | None = None


@dataclass
class ActorInputs:
    raw: dict
    sided: dict
    reliable: dict[str, set[int]]
    known: dict[str, set[int]]
    paths: list[Path]
    times: dict[tuple[str, int], float]


def _frame(value: str) -> int:
    return int("".join(c for c in value if c.isdigit()))


def _read(path: Path, sided: bool, *, times: dict | None = None) -> dict:
    manifest = json.loads(coordinate_manifest_path(path).read_text())
    # These source records explicitly carry native columns; do not silently use
    # legacy half-resolution boxes or invent a conversion without its binding.
    if manifest.get("image_size") != {"width": 1920, "height": 1080}:
        raise ValueError("native actor witness requires the declared native image size")
    output = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(NATIVE_COLUMNS) <= set(reader.fieldnames or []):
            raise ValueError("native actor witness requires explicit native box coordinates")
        for row in reader:
            side = row.get("side")
            if sided and side not in {"near", "far"}:
                continue
            box = tuple(float(row[c]) for c in NATIVE_COLUMNS)
            conf = float(row["conf"])
            if (
                not all(math.isfinite(v) for v in (*box, conf))
                or box[2] <= box[0]
                or box[3] <= box[1]
            ):
                continue
            key = (row["clip"], _frame(row["frame"]))
            if times is not None:
                time = float(row["t"])
                if not math.isfinite(time) or (key in times and times[key] != time):
                    raise ValueError("conflicting original native actor frame timestamps")
                times[key] = time
            if sided:
                key += (side,)
            obs = NativeBox(box, conf, row.get("track_id") if sided else None)
            # Exact repeated rows are one observation. Different overlapping boxes
            # remain alternatives and can cause the uniqueness test to abstain.
            values = output.setdefault(key, [])
            if obs not in values:
                values.append(obs)
    return output


def load_actor_inputs(match_dir: Path, sided_path: Path) -> ActorInputs:
    raw_path = sided_path.with_name(sided_path.name.replace("_sided_v1.csv", "_v1.csv"))
    court = match_dir / "court_H_per_frame_v1.npz"
    paths = [
        raw_path,
        coordinate_manifest_path(raw_path),
        sided_path,
        coordinate_manifest_path(sided_path),
        court,
    ]
    if not all(p.exists() for p in paths):
        raise ValueError("native actor witness missing original raw/sided/camera input")
    reliable, known = {}, {}
    with np.load(court, allow_pickle=False) as data:
        if data["reliable"].dtype != np.dtype(bool):
            raise ValueError("native actor camera reliability must be boolean")
        for clip, frame, accepted, H in zip(
            data["clips"], data["frames"], data["reliable"], data["H"], strict=True
        ):
            clip, frame = str(clip), int(frame)
            known.setdefault(clip, set()).add(frame)
            if bool(accepted) and np.isfinite(H).all():
                reliable.setdefault(clip, set()).add(frame)
    times = {}
    raw = _read(raw_path, False, times=times)
    return ActorInputs(raw, _read(sided_path, True), reliable, known, paths, times)


def _next(previous: NativeBox, options: list[NativeBox]) -> NativeBox | None:
    candidates = []
    height = previous.box[3] - previous.box[1]
    for candidate in options:
        ratio = (candidate.box[3] - candidate.box[1]) / height
        overlap = iou(previous.box, candidate.box)
        if candidate.confidence >= MIN_CONFIDENCE and 0.5 <= ratio <= 2.0 and overlap >= MIN_IOU:
            candidates.append((overlap, candidate))
    candidates.sort(key=lambda v: v[0], reverse=True)
    if not candidates or (
        len(candidates) > 1 and candidates[0][0] - candidates[1][0] < MIN_IOU_MARGIN
    ):
        return None
    return candidates[0][1]


def bridge_gap(
    inputs: ActorInputs, clip: str, start: int, end: int, *, shots: list[dict], fps: float
) -> dict:
    report = dict(
        start_frame=start,
        end_frame=end,
        supported=False,
        observations=[],
        coordinate_role="observed_native_2d_actor_box",
        court_coordinates=None,
        camera_reliability_upgraded=False,
        competitive_phase_inferred=False,
    )

    def hold(reason):
        return report | {"reason": reason}

    if not math.isfinite(fps) or fps <= 0 or end < start or end - start + 1 > fps * MAX_GAP_SECONDS:
        return hold("gap_not_short_and_bounded")
    left, right = start - 1, end + 1
    if (
        len(
            [
                s
                for s in shots
                if s.get("is_play_camera") is True
                and s["start_frame"] <= left
                and right <= s["end_frame"]
            ]
        )
        != 1
    ):
        return hold("gap_crosses_or_lacks_one_original_play_shot")
    reliable = inputs.reliable.get(clip, set())
    if (
        left not in reliable
        or right not in reliable
        or any(f in reliable for f in range(start, end + 1))
    ):
        return hold("not_one_bracketed_metric_camera_gap")
    if not set(range(left, right + 1)) <= inputs.known.get(clip, set()):
        return hold("missing_native_camera_frame_inventory")
    times = [inputs.times.get((clip, f)) for f in range(left, right + 1)]
    # players.detection_row rounds source timestamps to centiseconds. Two
    # rounded endpoints can differ by up to 10ms. Check anchored elapsed time,
    # not independent adjacent deltas that could conceal a wrong native cadence.
    if (
        any(t is None for t in times)
        or any(b < a for a, b in zip(times, times[1:]))
        or any(abs((t - times[0]) - i / fps) > 0.010001 for i, t in enumerate(times))
    ):
        return hold("native_actor_clock_not_frame_adjacent")
    side_paths = {}
    for side in ("near", "far"):
        a, b = inputs.sided.get((clip, left, side), []), inputs.sided.get((clip, right, side), [])
        if len(a) != 1 or len(b) != 1 or not a[0].track_id or a[0].track_id != b[0].track_id:
            return hold("missing_or_conflicting_reliable_identity_anchors")
        paths = []
        for anchor, sequence, endpoint in (
            (a[0], range(start, right + 1), b[0]),
            (b[0], range(end, left - 1, -1), a[0]),
        ):
            previous, path = anchor, {}
            for frame in sequence:
                selected = _next(previous, inputs.raw.get((clip, frame), []))
                if selected is None:
                    return hold("missing_or_ambiguous_frame_adjacent_native_match")
                path[frame] = selected
                previous = selected
            if previous.box != endpoint.box:
                return hold("native_path_disagrees_with_opposite_anchor")
            paths.append(path)
        if any(paths[0][f].box != paths[1][f].box for f in range(start, end + 1)):
            return hold("forward_backward_native_identity_disagreement")
        side_paths[side] = paths[0]
    for frame in range(start, end + 1):
        near, far = (side_paths[side][frame] for side in ("near", "far"))
        if iou(near.box, far.box) >= MIN_IOU:
            return hold("two_actor_paths_overlap")
        report["observations"].append(
            dict(
                frame=frame,
                near_box=list(near.box),
                far_box=list(far.box),
                near_confidence=near.confidence,
                far_confidence=far.confidence,
            )
        )
    return report | {
        "supported": True,
        "reason": "unique_bidirectional_observed_boxes",
        "anchor_frames": [left, right],
    }


def witnesses(
    inputs: ActorInputs, clip: str, requested: set[int], *, shots: list[dict], fps: float
) -> list[dict]:
    """Use full original camera gaps, never just a favorable subwindow of one."""
    known = sorted(inputs.known.get(clip, set()))
    reliable = inputs.reliable.get(clip, set())
    gaps, run = [], []
    for f in known:
        if f in reliable or (run and f != run[-1] + 1):
            if run:
                gaps.append(run)
            run = []
        if f not in reliable:
            run.append(f)
    if run:
        gaps.append(run)
    return [
        bridge_gap(inputs, clip, g[0], g[-1], shots=shots, fps=fps)
        for g in gaps
        if requested.intersection(g)
    ]
