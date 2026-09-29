"""Compose the coarse full-frame ball track with the native-crop detections.

Two modes.

Default: the local-crop detector is a recovery proposal layer only. Every point in the base
track is frozen with score 1.0, so candidates are decoded only into missing frames, and
recovered runs are retained only when the surrounding observations support a low-residual
piecewise-linear trajectory with one allowed tennis kink.

``--crop-authoritative``: wherever a crop detection sits within ``--crop-authority-px1080``
of the coarse lock it replaces it, and the coarse pass only proposes fills for frames the
crop has nothing for. The motivation is sound -- the full-frame pass warps 1920x1080 down to
512x288 and lands 99.2% of its output on a 3.75-native-pixel grid, while the crop pass sees
real sensor pixels 1:1 -- and a truth-aware pick among the crop's five peaks does beat the
composed track (1.45 vs 2.25 native px median on the owner development windows). But no
selection rule available here can find that pick: taking the highest-scoring or most-agreed
crop peak makes the composed track worse on both frozen cohorts, so this stays opt-in until
a learned selector exists.

Both modes keep the run-level acceptance tests (trajectory fit, speed safety) on whatever is
filled. In crop-authoritative mode each output row also carries a ``provenance:`` token in
``sources`` -- ``crop``, ``coarse``, ``far_native`` or ``interpolated`` -- appended to the
existing source list so downstream can tell real native evidence from a downscaled peak or a
decoder interpolation without re-deriving it from file names.
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
    Candidate,
    DecoderConfig,
    config_for_consumer,
    merge_candidates,
    select_consistent_path,
)
from cv.pipeline.track_artifact import write_track_artifact
from cv.pipeline.event_refine import refine_anchor


CROP_FAMILY = "crop"
COARSE_FAMILY = "coarse"
FAR_NATIVE_FAMILY = "far_native"


def candidate_family(path: Path, header: list[str]) -> str:
    """Classify one candidate file by the evidence it was decoded from.

    ``crop_provenance`` is written only by :mod:`ball_local_refine`, so its presence is the
    reliable marker of a true-native crop pass; the far-native tiling pass is identified by
    its output tag because it shares the full-frame schema.
    """
    if "crop_provenance" in header:
        return CROP_FAMILY
    if "far_native" in path.stem:
        return FAR_NATIVE_FAMILY
    return COARSE_FAMILY


def _provenance(sources: tuple[str, ...] | list[str], families: dict[str, str]) -> str:
    resolved = {families[source] for source in sources if source in families}
    for family in (CROP_FAMILY, FAR_NATIVE_FAMILY):
        if family in resolved:
            return family
    if any(source == "interpolated" for source in sources):
        return "interpolated"
    return COARSE_FAMILY


def apply_crop_authority(
    base_rows: dict[int, dict],
    crop_candidates: dict[int, list[Candidate]],
    gate_px: float,
) -> tuple[dict[int, dict], int]:
    """Let a nearby native-crop detection overwrite the coarse position it was centred on.

    ``gate_px`` bounds how far a replacement may move the lock, in the same units the track
    is written in. Within that radius the crop detection with the most agreeing detectors
    wins, then the lowest rank, then the highest score.
    """
    corrected: dict[int, dict] = {}
    replaced = 0
    for frame, row in base_rows.items():
        eligible = [
            candidate
            for candidate in crop_candidates.get(frame, ())
            if math.hypot(candidate.x - float(row["x"]), candidate.y - float(row["y"])) <= gate_px
        ]
        if not eligible:
            corrected[frame] = row
            continue
        best = max(eligible, key=lambda item: (len(item.sources), -item.rank, item.score))
        replaced += 1
        corrected[frame] = {
            **row,
            "x": best.x,
            "y": best.y,
            "score": best.score,
            "rank": best.rank,
            "sources": "+".join(
                filter(None, (row.get("sources", ""), "crop_authoritative", *best.sources))
            ),
        }
    return corrected, replaced


def _frame_number(value: str) -> int:
    match = re.search(r"\d+", value)
    if match is None:
        raise ValueError(f"frame has no numeric index: {value}")
    return int(match.group())


def _runs(frames: list[int]) -> list[tuple[int, int]]:
    if not frames:
        return []
    output = []
    start = previous = frames[0]
    for frame in frames[1:]:
        if frame != previous + 1:
            output.append((start, previous))
            start = frame
        previous = frame
    output.append((start, previous))
    return output


def fit_supported_runs(
    rows_by_frame: dict[int, dict],
    local_frames: list[int],
    fps: float,
    minimum_run: int = 2,
    maximum_rmse: float = 6.0,
    context_seconds: float = 0.32,
) -> set[int]:
    context = max(4, round(context_seconds * fps))
    supported = set()
    for start, end in _runs(sorted(local_frames)):
        run_length = end - start + 1
        if run_length < minimum_run:
            continue
        rows = [
            row
            for frame, row in rows_by_frame.items()
            if start - context <= frame <= end + context
            and (not row.get("sources", "").startswith("local_gap_fill") or start <= frame <= end)
        ]
        rows.sort(key=lambda row: _frame_number(row["frame"]))
        if len(rows) < 5:
            continue
        frames = np.asarray([_frame_number(row["frame"]) for row in rows], dtype=float)
        values_x = np.asarray([float(row["x"]) for row in rows])
        values_y = np.asarray([float(row["y"]) for row in rows])
        half_run = run_length / 2.0
        result = refine_anchor(
            frames,
            values_x,
            values_y,
            hint=(start + end) / 2.0,
            search=half_run + max(2.0, 6.0 * fps / 50.0),
            window=half_run + context,
            step=0.2,
        )
        if result.ok and result.rmse <= maximum_rmse:
            supported.update(range(start, end + 1))
    return supported


def speed_safe_frames(
    rows_by_frame: dict[int, dict],
    supported_frames: set[int],
    fps: float,
    maximum_speed_px_s: float = 4000.0,
) -> set[int]:
    safe = set()
    observed = sorted(rows_by_frame)
    for start, end in _runs(sorted(supported_frames)):
        run_frames = list(range(start, end + 1))
        prior = [frame for frame in observed if frame < start and frame not in supported_frames]
        following = [frame for frame in observed if frame > end and frame not in supported_frames]
        check_frames = [
            *([prior[-1]] if prior else []),
            *run_frames,
            *([following[0]] if following else []),
        ]
        unsafe = False
        for left_frame, right_frame in zip(check_frames, check_frames[1:]):
            if left_frame not in rows_by_frame or right_frame not in rows_by_frame:
                continue
            left = rows_by_frame[left_frame]
            right = rows_by_frame[right_frame]
            speed = (
                math.hypot(
                    float(right["x"]) - float(left["x"]),
                    float(right["y"]) - float(left["y"]),
                )
                * fps
                / (right_frame - left_frame)
            )
            if speed >= maximum_speed_px_s:
                unsafe = True
                break
        if not unsafe:
            safe.update(run_frames)
    return safe


def recovery_config_for_fps(fps: float) -> DecoderConfig:
    config = config_for_consumer(fps, "availability")
    return DecoderConfig(
        **{
            **config.__dict__,
            "rank_cost": 0.01,
            "score_cost": 0.25,
            "kink_cost": 0.35,
            "missing_cost": 1.2,
            "restart_cost": 0.2,
        }
    )


def one_sided_extensions(
    base_rows: dict[int, dict],
    local_candidates: dict[int, list[Candidate]],
    fps: float,
    maximum_projection_error: float = 12.0,
    maximum_fit_rmse: float = 3.0,
    maximum_opposite_speed_px_s: float = 4000.0,
) -> dict[int, Candidate]:
    context = max(4, round(8 * fps / 50.0))
    recovered = {}
    for direction in (-1, 1):
        for frame, candidates in local_candidates.items():
            if frame in base_rows or frame - direction not in base_rows:
                continue
            context_frames = [frame - direction * offset for offset in range(1, context + 1)]
            if any(context_frame not in base_rows for context_frame in context_frames):
                continue
            ordered = sorted(context_frames)
            values_x = np.asarray([float(base_rows[index]["x"]) for index in ordered])
            values_y = np.asarray([float(base_rows[index]["y"]) for index in ordered])
            frames = np.asarray(ordered, dtype=float)
            predicted = []
            residuals = []
            for values in (values_x, values_y):
                coefficients = np.polyfit(frames, values, 1)
                fitted = np.polyval(coefficients, frames)
                residuals.append(values - fitted)
                predicted.append(float(np.polyval(coefficients, frame)))
            fit_rmse = float(np.sqrt(np.mean(np.square(residuals[0]) + np.square(residuals[1]))))
            if fit_rmse > maximum_fit_rmse:
                continue
            eligible = [
                candidate
                for candidate in candidates
                if any("trajectory" in source for source in candidate.sources)
                and math.hypot(candidate.x - predicted[0], candidate.y - predicted[1])
                <= maximum_projection_error
            ]
            if not eligible:
                continue
            chosen = min(
                eligible,
                key=lambda candidate: (
                    candidate.rank,
                    math.hypot(
                        candidate.x - predicted[0],
                        candidate.y - predicted[1],
                    ),
                ),
            )
            opposite_frames = [
                base_frame for base_frame in base_rows if (base_frame - frame) * direction > 0
            ]
            if opposite_frames:
                opposite_frame = min(
                    opposite_frames,
                    key=lambda base_frame: abs(base_frame - frame),
                )
                if abs(opposite_frame - frame) <= context:
                    opposite = base_rows[opposite_frame]
                    opposite_speed = (
                        math.hypot(
                            chosen.x - float(opposite["x"]),
                            chosen.y - float(opposite["y"]),
                        )
                        * fps
                        / abs(opposite_frame - frame)
                    )
                    if opposite_speed >= maximum_opposite_speed_px_s:
                        continue
            prior = recovered.get(frame)
            if prior is None or chosen.rank < prior.rank:
                recovered[frame] = Candidate(
                    chosen.x,
                    chosen.y,
                    chosen.score,
                    chosen.rank,
                    (*chosen.sources, "one_sided_extension"),
                )
    return recovered


def augment_track(
    base_path: Path,
    candidate_paths: Path | list[Path],
    output_path: Path,
    fps: float,
    minimum_run: int = 2,
    maximum_rmse: float = 6.0,
    context_seconds: float = 0.32,
    crop_authoritative: bool = False,
    crop_authority_px: float = 24.0,
) -> dict[str, int]:
    base: dict[str, dict[int, dict]] = defaultdict(dict)
    with base_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            base[row["clip"]][_frame_number(row["frame"])] = row

    local: dict[str, dict[int, list[tuple[str, Candidate]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    families: dict[str, str] = {}
    paths = [candidate_paths] if isinstance(candidate_paths, Path) else candidate_paths
    for candidate_path in paths:
        with candidate_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            family = candidate_family(candidate_path, list(reader.fieldnames or ()))
            for row in reader:
                provenance = row.get("crop_provenance", "local_crop")
                source = f"{candidate_path.stem}:{provenance}"
                families[source] = family
                local[row["clip"]][_frame_number(row["frame"])].append(
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
    recovery_config = recovery_config_for_fps(fps)
    output = []
    proposed = 0
    retained = 0
    accepted_runs = 0
    endpoint_extensions = 0
    crop_corrections = 0
    for clip, base_rows in sorted(base.items()):
        clip_local = local.get(clip, {})
        clip_crop = {
            frame: merge_candidates(
                [item for item in candidates if families[item[0]] != COARSE_FAMILY],
                config,
            )
            for frame, candidates in clip_local.items()
        }
        clip_crop = {frame: merged for frame, merged in clip_crop.items() if merged}
        if crop_authoritative:
            base_rows, replaced = apply_crop_authority(base_rows, clip_crop, crop_authority_px)
            crop_corrections += replaced
        frames: dict[int, list[Candidate]] = defaultdict(list)
        for frame, row in base_rows.items():
            frames[frame].append(
                Candidate(
                    float(row["x"]),
                    float(row["y"]),
                    1.0,
                    0,
                    ("base_a", "base_b"),
                )
            )
        for frame, candidates in clip_local.items():
            # Crop-authoritative: the coarse pass only proposes fills where the native crop
            # has nothing to say about the frame at all.
            if crop_authoritative and frame in clip_crop:
                frames[frame].extend(clip_crop[frame])
            else:
                frames[frame].extend(merge_candidates(candidates, config))
        selected = select_consistent_path(dict(frames), fps, recovery_config)
        fills = {
            frame: candidate
            for frame, candidate in selected
            if candidate is not None and frame not in base_rows
        }
        endpoint_fills = one_sided_extensions(
            base_rows,
            {
                frame: [candidate for _, candidate in candidates]
                for frame, candidates in local.get(clip, {}).items()
            },
            fps,
        )
        fills.update(endpoint_fills)
        endpoint_extensions += len(endpoint_fills)
        proposed += len(fills)
        combined = dict(base_rows)
        for frame, candidate in fills.items():
            combined[frame] = {
                "clip": clip,
                "frame": f"f_{frame:04d}.jpg",
                "x": candidate.x,
                "y": candidate.y,
                "track_id": 0,
                "score": candidate.score,
                "rank": candidate.rank,
                "sources": f"local_gap_fill+{'+'.join(candidate.sources)}",
            }
        supported = fit_supported_runs(
            combined,
            [frame for frame in fills if frame not in endpoint_fills],
            fps,
            minimum_run,
            maximum_rmse,
            context_seconds,
        )
        supported.update(endpoint_fills)
        supported = speed_safe_frames(combined, supported, fps)
        accepted_runs += len(_runs(sorted(supported)))
        retained += len(supported)
        for frame in sorted(set(base_rows) | supported):
            row = combined[frame]
            sources = row.get("sources", "base")
            if crop_authoritative:
                tokens = tuple(filter(None, sources.split("+")))
                sources = f"{sources}+provenance:{_provenance(tokens, families)}"
            output.append(
                {
                    "clip": clip,
                    "frame": row["frame"],
                    "x": row["x"],
                    "y": row["y"],
                    "track_id": row.get("track_id", 0),
                    "score": row.get("score", ""),
                    "rank": row.get("rank", ""),
                    "sources": sources,
                }
            )
    write_track_artifact(output_path, output, [base_path, *paths])
    return {
        "clips": len(base),
        "base_points": sum(len(rows) for rows in base.values()),
        "proposed_local_points": proposed,
        "retained_local_points": retained,
        "accepted_runs": accepted_runs,
        "endpoint_extensions": endpoint_extensions,
        "crop_authoritative": crop_authoritative,
        "crop_corrected_base_points": crop_corrections,
        "output_points": len(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--minimum-run", type=int, default=2)
    parser.add_argument("--maximum-rmse", type=float, default=6.0)
    parser.add_argument("--context-seconds", type=float, default=0.32)
    parser.add_argument(
        "--crop-authoritative",
        action="store_true",
        help="let a native-crop detection within --crop-authority-px1080 of the coarse lock "
        "replace it, and restrict coarse fills to frames the crop has nothing for",
    )
    parser.add_argument(
        "--crop-authority-px1080",
        type=float,
        default=4.0,
        help="native-pixel radius for --crop-authoritative replacement; beyond about 4 "
        "native px a replacement was measured to be a wrong-object jump almost every time",
    )
    args = parser.parse_args()
    contract = res.read_coordinate_manifest(args.base)
    artifact_size = (
        res.manifest_artifact_size(contract) if contract else res.CANONICAL_SIZE
    )
    crop_authority_px = float(
        res.scale_points(
            np.asarray([args.crop_authority_px1080, 0.0]),
            res.NATIVE_SIZE,
            artifact_size,
        )[0]
    )
    print(
        augment_track(
            args.base,
            args.candidates,
            args.output,
            args.fps,
            args.minimum_run,
            args.maximum_rmse,
            args.context_seconds,
            args.crop_authoritative,
            crop_authority_px,
        )
    )


if __name__ == "__main__":
    main()
