"""Replace short false-lock islands only when candidate evidence bridges both sides.

Measured state: across 3,586 gate records this stage has never replaced a point, and it
replaced none on either wk1_s4 cohort. ``--passthrough`` makes it an explicit no-op that
still writes the artifact the composer's next stage reads, so the stage can be switched off
without changing any downstream path.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from cv.pipeline import resolution as res
from cv.pipeline.ball_track_consensus import (
    CONSENSUS_PIXEL_SPACE,
    Candidate,
    config_for_consumer,
    merge_candidates,
)
from cv.pipeline.ball_track_local_augment import _runs
from cv.pipeline.event_refine import refine_anchor
from cv.pipeline.track_artifact import write_track_artifact


def _frame_number(value: str) -> int:
    match = re.search(r"\d+", value)
    if match is None:
        raise ValueError(f"frame has no numeric index: {value}")
    return int(match.group())


def find_detour_replacements(
    base_rows: dict[int, dict],
    candidates_by_frame: dict[int, list[Candidate]],
    fps: float,
    maximum_run_seconds: float = 0.18,
    maximum_boundary_gap_seconds: float = 0.16,
    minimum_detour_px: float = 60.0,
    maximum_bridge_error_px: float = 45.0,
    maximum_fit_rmse: float = 6.0,
    minimum_agreement_fraction: float = 0.5,
) -> dict[int, Candidate]:
    """Return replacements for complete, short islands that make a transient detour."""
    observed = sorted(base_rows)
    maximum_run = max(2, round(maximum_run_seconds * fps))
    maximum_boundary_gap = max(2, round(maximum_boundary_gap_seconds * fps))
    replacements = {}
    for start, end in _runs(observed):
        run_frames = list(range(start, end + 1))
        if len(run_frames) > maximum_run:
            continue
        prior = [frame for frame in observed if frame < start]
        following = [frame for frame in observed if frame > end]
        if not prior or not following:
            continue
        left_frame = prior[-1]
        right_frame = following[0]
        if start - left_frame > maximum_boundary_gap or right_frame - end > maximum_boundary_gap:
            continue
        left = np.asarray([float(base_rows[left_frame]["x"]), float(base_rows[left_frame]["y"])])
        right = np.asarray([float(base_rows[right_frame]["x"]), float(base_rows[right_frame]["y"])])
        span = right_frame - left_frame

        def bridge(frame: int) -> np.ndarray:
            fraction = (frame - left_frame) / span
            return left * (1.0 - fraction) + right * fraction

        current_errors = [
            float(
                np.linalg.norm(
                    np.asarray(
                        [
                            float(base_rows[frame]["x"]),
                            float(base_rows[frame]["y"]),
                        ]
                    )
                    - bridge(frame)
                )
            )
            for frame in run_frames
        ]
        if float(np.median(current_errors)) < minimum_detour_px:
            continue
        alternate = {}
        for frame in run_frames:
            candidates = candidates_by_frame.get(frame, [])
            if not candidates:
                break
            chosen = min(
                candidates,
                key=lambda candidate: math.dist(
                    (candidate.x, candidate.y),
                    bridge(frame),
                ),
            )
            if math.dist((chosen.x, chosen.y), bridge(frame)) > maximum_bridge_error_px:
                break
            alternate[frame] = chosen
        if len(alternate) != len(run_frames):
            continue
        agreement_fraction = sum(
            len(candidate.sources) >= 2 for candidate in alternate.values()
        ) / len(alternate)
        if agreement_fraction < minimum_agreement_fraction:
            continue
        frames = np.asarray([left_frame, *run_frames, right_frame], dtype=float)
        points = np.asarray(
            [
                left,
                *[np.asarray([alternate[frame].x, alternate[frame].y]) for frame in run_frames],
                right,
            ]
        )
        fit = refine_anchor(
            frames,
            points[:, 0],
            points[:, 1],
            hint=(start + end) / 2.0,
            search=max(2.0, len(run_frames)),
            window=right_frame - left_frame,
            step=0.1,
        )
        if not fit.ok or fit.rmse > maximum_fit_rmse:
            continue
        replacements.update(alternate)
    return replacements


def repair_csv(
    base_path: Path,
    candidate_paths: list[Path],
    output_path: Path,
    fps: float,
    enabled: bool = True,
) -> dict[str, int]:
    base: dict[str, dict[int, dict]] = defaultdict(dict)
    res.require_artifact_space(base_path, CONSENSUS_PIXEL_SPACE, consumer="detour repair")
    with base_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            base[row["clip"]][_frame_number(row["frame"])] = row
    raw_candidates = defaultdict(lambda: defaultdict(list))
    for path in candidate_paths:
        source = path.stem
        res.require_artifact_space(path, CONSENSUS_PIXEL_SPACE, consumer="detour repair")
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                raw_candidates[row["clip"]][_frame_number(row["frame"])].append(
                    (
                        source,
                        Candidate(
                            float(row["x"]),
                            float(row["y"]),
                            float(row["score"]),
                            int(row.get("rank", 0)),
                            (source,),
                        ),
                    )
                )
    config = config_for_consumer(fps, "availability")
    output = []
    replacement_count = 0
    repaired_runs = 0
    for clip, rows in sorted(base.items()):
        if enabled:
            candidates = {
                frame: merge_candidates(frame_candidates, config)
                for frame, frame_candidates in raw_candidates[clip].items()
            }
            replacements = find_detour_replacements(rows, candidates, fps)
        else:
            replacements = {}
        replacement_count += len(replacements)
        repaired_runs += len(_runs(sorted(replacements)))
        for frame, row in sorted(rows.items()):
            replacement = replacements.get(frame)
            output.append(
                {
                    "clip": clip,
                    "frame": row["frame"],
                    "x": replacement.x if replacement is not None else row["x"],
                    "y": replacement.y if replacement is not None else row["y"],
                    "track_id": row.get("track_id", 0),
                    "score": replacement.score if replacement is not None else row.get("score", ""),
                    "rank": replacement.rank if replacement is not None else row.get("rank", ""),
                    "sources": (
                        f"transient_detour_replacement+{'+'.join(replacement.sources)}"
                        if replacement is not None
                        else row.get("sources", "base")
                    ),
                }
            )
    write_track_artifact(output_path, output, [base_path, *candidate_paths])
    return {
        "clips": len(base),
        "enabled": enabled,
        "base_points": sum(len(rows) for rows in base.values()),
        "repaired_runs": repaired_runs,
        "replaced_points": replacement_count,
        "output_points": len(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument(
        "--passthrough",
        dest="enabled",
        action="store_false",
        help="write the base track through unchanged; the repair has never fired on any "
        "measured cohort",
    )
    args = parser.parse_args()
    print(repair_csv(args.base, args.candidates, args.output, args.fps, args.enabled))


if __name__ == "__main__":
    main()
