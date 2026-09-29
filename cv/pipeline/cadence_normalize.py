"""Normalize regular duplicate-frame conversions and reject irregular timebases."""

from __future__ import annotations

import bisect
import csv
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

STANDARD_FPS = (24000 / 1001, 24.0, 25.0, 30.0, 50.0, 60000 / 1001, 60.0)
MAX_TARGET_FPS_ERROR = 0.75
# A regular conversion is a rational repeat pattern, not necessarily a stream of
# isolated duplicate pairs.  For example 24 -> 60 has 2:3 exposure holds, so half
# of its duplicate groups have three source frames.  The old pair/gap-share checks
# consequently accepted that source as raw 60 fps.
MAX_PATTERN_PHASE_ERROR_FRAMES = 0.55


def cadence_action(row: dict) -> dict:
    if row["decision"] == "timing_usable":
        return {"action": "retain_native", "target_fps": float(row["fps"]), "reasons": []}

    source_fps = float(row["fps"])
    effective_fps = float(row["effective_unique_fps"])
    target_fps = min(STANDARD_FPS, key=lambda value: abs(value - effective_fps))
    groups = row.get("equivalence_groups", [])
    exposure_lengths = _exposure_lengths(int(row["frame_count"]), groups)
    pair_share = sum(length == 2 for length in exposure_lengths) / max(1, len(exposure_lengths))
    starts = [int(value) for value in row.get("duplicate_pair_starts", [])]
    gaps = [right - left for left, right in zip(starts, starts[1:])]
    gap_counts = Counter(gaps)
    period, period_count = gap_counts.most_common(1)[0] if gap_counts else (0, 0)
    periodic_share = period_count / max(1, len(gaps))
    fps_error = abs(target_fps - effective_fps)
    pattern = _standard_pattern(exposure_lengths, source_fps, target_fps)
    regular = fps_error <= MAX_TARGET_FPS_ERROR and pattern["consistent"]
    if regular:
        return {
            "action": "heal_regular_duplicate_cadence",
            "target_fps": target_fps,
            "source_fps": source_fps,
            "period": period,
            "periodic_gap_share": periodic_share,
            "pair_group_share": pair_share,
            "effective_unique_fps": effective_fps,
            "pattern_phase_error_frames": pattern["phase_error_frames"],
            "pattern_lengths": pattern["lengths"],
            "reasons": [],
        }
    return {
        "action": "discard_invalid_timebase",
        "target_fps": None,
        "source_fps": source_fps,
        "period": period,
        "periodic_gap_share": periodic_share,
        "pair_group_share": pair_share,
        "effective_unique_fps": effective_fps,
        "pattern_phase_error_frames": pattern["phase_error_frames"],
        "pattern_lengths": pattern["lengths"],
        "reasons": ["irregular_duplicate_cadence"],
    }


def _exposure_lengths(frame_count: int, groups: list[list[int]]) -> list[int]:
    """Return source-frame holds, including single-frame unique exposures."""
    by_first = {int(sorted(group)[0]): sorted({int(frame) for frame in group}) for group in groups}
    lengths: list[int] = []
    frame = 1
    while frame <= frame_count:
        group = by_first.get(frame)
        if group is None:
            lengths.append(1)
            frame += 1
            continue
        if group != list(range(frame, group[-1] + 1)):
            # A non-contiguous equivalence group cannot describe an exposure hold.
            return []
        lengths.append(len(group))
        frame = group[-1] + 1
    return lengths


def _standard_pattern(lengths: list[int], source_fps: float, target_fps: float) -> dict:
    """Check a duplicate sequence against a standard rational exposure clock.

    The cumulative boundary of every exposure must remain within half a source frame
    of ``source_fps / target_fps``.  That accepts 2:3, 4:5, and 1:2 holds while
    rejecting a random mixture with the same aggregate effective frame rate.
    """
    if not lengths or target_fps <= 0 or source_fps <= target_fps:
        return {"consistent": False, "phase_error_frames": None, "lengths": []}
    ratio = source_fps / target_fps
    allowed = {math.floor(ratio), math.ceil(ratio)}
    # Extraction spans may end midway through a held exposure.  Ignore that lone
    # tail when the nominal hold is at least two frames; it contains no evidence
    # of an irregular cadence (e.g. a 50 fps clip ending after a single frame of
    # its final 25 fps exposure).
    complete_lengths = lengths
    if len(lengths) > 1 and lengths[-1] < min(allowed):
        complete_lengths = lengths[:-1]
    if min(allowed) < 1 or any(length not in allowed for length in complete_lengths):
        return {"consistent": False, "phase_error_frames": None, "lengths": sorted(set(lengths))}
    boundaries = [0]
    for length in complete_lengths:
        boundaries.append(boundaries[-1] + length)
    residuals = [boundary - index * ratio for index, boundary in enumerate(boundaries)]
    phase = sum(residuals) / len(residuals)
    phase_error = max(abs(residual - phase) for residual in residuals)
    return {
        "consistent": phase_error <= MAX_PATTERN_PHASE_ERROR_FRAMES,
        "phase_error_frames": round(phase_error, 6),
        "lengths": sorted(set(lengths)),
    }


def observation_timeline(row: dict) -> list[dict]:
    frame_count = int(row["frame_count"])
    source_fps = float(row["fps"])
    group_by_frame = {}
    for source_group in row.get("equivalence_groups", []):
        group = tuple(sorted({int(frame) for frame in source_group}))
        for frame in group:
            group_by_frame[frame] = group

    observations = []
    consumed = set()
    for source_frame in range(1, frame_count + 1):
        if source_frame in consumed:
            continue
        group = group_by_frame.get(source_frame, (source_frame,))
        consumed.update(group)
        first, last = group[0], group[-1]
        observations.append(
            {
                "observation_frame": len(observations) + 1,
                "canonical_source_frame": first,
                "source_frames": list(group),
                "time_lower_seconds": (first - 1) / source_fps,
                "time_upper_seconds": last / source_fps,
                "time_mid_seconds": ((first - 1) + last) / (2.0 * source_fps),
            }
        )
    return observations


def source_frame_to_observation(frame: float, timeline: list[dict], source_fps: float) -> float:
    rounded = int(round(frame))
    if math.isclose(frame, rounded, abs_tol=1e-6):
        for observation in timeline:
            if rounded in observation["source_frames"]:
                return float(observation["observation_frame"])

    time_seconds = (float(frame) - 1.0) / source_fps
    centers = [float(row["time_mid_seconds"]) for row in timeline]
    right = bisect.bisect_left(centers, time_seconds)
    if right <= 0:
        value = 1.0
    elif right >= len(centers):
        value = float(len(centers))
    else:
        left = right - 1
        span = centers[right] - centers[left]
        fraction = (time_seconds - centers[left]) / span if span > 0 else 0.0
        value = float(left + 1) + fraction
    return round(value * 2.0) / 2.0


def materialize_observations(
    source_frames: Path,
    destination_frames: Path,
    row: dict,
) -> list[dict]:
    timeline = observation_timeline(row)
    destination_frames.mkdir(parents=True, exist_ok=True)
    for observation in timeline:
        source = source_frames / f"f_{observation['canonical_source_frame']:04d}.jpg"
        destination = destination_frames / f"f_{observation['observation_frame']:04d}.jpg"
        if not source.exists():
            raise FileNotFoundError(source)
        destination.symlink_to(source.resolve())
    (destination_frames / "observation_timeline.json").write_text(
        json.dumps(
            {
                "schema": "cadence_observation_timeline_v1",
                "match_id": row["match_id"],
                "clip": row["clip"],
                "source_fps": float(row["fps"]),
                "target_fps": cadence_action(row)["target_fps"],
                "observations": timeline,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return timeline


def remap_track(
    source: Path,
    destination: Path,
    timelines: dict[str, list[dict]],
) -> None:
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    observation_by_clip_source = {
        clip: {
            int(frame): int(item["observation_frame"])
            for item in timeline
            for frame in item["source_frames"]
        }
        for clip, timeline in timelines.items()
    }
    selected = {}
    for row in rows:
        clip = row["clip"]
        observation_by_source = observation_by_clip_source.get(clip)
        if observation_by_source is None:
            continue
        source_frame = int(Path(row["frame"]).stem.removeprefix("f_"))
        observation = observation_by_source[source_frame]
        candidate = dict(row)
        candidate["frame"] = f"f_{observation:04d}.jpg"
        key = (clip, observation)
        score = float(candidate.get("score") or 0.0)
        incumbent_score = float(selected.get(key, {}).get("score") or 0.0)
        if key not in selected or score > incumbent_score:
            selected[key] = candidate
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected[key] for key in sorted(selected))


def reset_directory(path: Path) -> None:
    if path.is_symlink():
        target = path.resolve()
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        return
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def normalize_source_video(source: Path, destination: Path, target_fps: float) -> None:
    """Materialize only unique visual exposures on a regular cadence.

    ``mpdecimate`` removes repeated pictures; retiming the retained exposure sequence to
    the inferred standard rate keeps point timestamps usable by current frame-indexed
    consumers.  The source-time intervals and detection evidence stay in source_integrity.
    """
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            f"mpdecimate=hi=1280:lo=640:frac=0.5,setpts=N/({target_fps}*TB)",
            "-r",
            str(target_fps),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            "-y",
            str(destination),
        ],
        check=True,
    )


# --- Pattern-based native-cadence restore (``--cadence-restore``) -------------------------
#
# ``normalize_source_video`` above keeps whatever ``mpdecimate`` retains and renumbers it,
# so a genuinely static stretch (distinct captures that happen to look alike) is dropped and
# the clock after it is compressed.  The restore below instead fits the conversion clock:
# a native capture ``c`` is shown on absolute source frames ``j`` (0-based) with
#
#     c(j) = floor((j + phase) * native / source),
#
# so the repeats are *predicted* by (native/source, phase) and the detector only confirms
# them.  A predicted repeat that the detector sees as a new picture refutes the fit; an extra
# detected repeat (a still picture) is kept as the distinct capture it is.  Only isolated-pair
# conversions (1 < source/native <= 2, e.g. 25->50 doubling or 25->30 pulldown) qualify;
# 2:3 or irregular holds are not restored.
RESTORE_MAX_PERIOD = 12
RESTORE_MIN_PREDICTED_RECALL = 0.95
RESTORE_MIN_PHASE_MARGIN = 0.25
RESTORE_NATIVE_FPS = (24000 / 1001, 24.0, 25.0, 30000 / 1001, 30.0)
# Still pictures make a few long equivalence groups; a 2:3 conversion makes half of them long.
RESTORE_MAX_LONG_GROUP_SHARE = 0.1


def restore_ratio(source_fps: float, native_fps: float) -> tuple[int, int] | None:
    """Return (native, source) as small coprime integers, or None if not a pair conversion."""
    from fractions import Fraction

    ratio = Fraction(native_fps / source_fps).limit_denominator(RESTORE_MAX_PERIOD)
    if not math.isclose(float(ratio), native_fps / source_fps, rel_tol=1e-4):
        return None
    if not (Fraction(1, 2) <= ratio < 1):
        return None
    return ratio.numerator, ratio.denominator


def capture_index(frame: int, phase: int, ratio: tuple[int, int]) -> int:
    """Native capture shown on absolute 0-based source frame ``frame``."""
    native, source = ratio
    return ((frame + phase) * native) // source


def predicted_repeats(first: int, count: int, phase: int, ratio: tuple[int, int]) -> set[int]:
    """Absolute source frames in ``[first, first + count)`` that repeat their predecessor."""
    return {
        frame
        for frame in range(first + 1, first + count)
        if capture_index(frame, phase, ratio) == capture_index(frame - 1, phase, ratio)
    }


def fit_restore_phase(
    *,
    first: int,
    count: int,
    observed_repeats: set[int],
    ratio: tuple[int, int],
) -> dict:
    """Score every phase of the conversion clock against detected repeats.

    ``observed_repeats`` are absolute 0-based frames the detector found equal to their
    predecessor.  Recall is the share of predicted repeats that were observed; a wrong phase
    predicts repeats on frames that carry new pictures and loses recall.
    """
    _, source = ratio
    scores = []
    for phase in range(source):
        predicted = predicted_repeats(first, count, phase, ratio)
        hits = len(predicted & observed_repeats)
        scores.append(
            {
                "phase": phase,
                "predicted": len(predicted),
                "confirmed": hits,
                "recall": hits / max(1, len(predicted)),
                "unexplained_repeats": len(observed_repeats - predicted),
            }
        )
    ranked = sorted(scores, key=lambda row: (-row["recall"], row["phase"]))
    best = ranked[0]
    runner_up = ranked[1]["recall"] if len(ranked) > 1 else 0.0
    return {
        "phase": best["phase"],
        "recall": round(best["recall"], 6),
        "predicted": best["predicted"],
        "confirmed": best["confirmed"],
        "unexplained_repeats": best["unexplained_repeats"],
        "phase_margin": round(best["recall"] - runner_up, 6),
        "scores": scores,
    }


def restore_native_fps_candidates(source_fps: float, effective_unique_fps: float) -> list[float]:
    """Isolated-pair native rates that can explain the unique-picture rate, slowest first.

    Still pictures only lower the observed unique rate, so the native rate is a
    pair-convertible standard rate not below it (less the detector tolerance).  Stills can
    push a 25 -> 30 pulldown below 24.75 unique pictures a second, so the slowest candidate
    is not necessarily the native one; the conversion-clock fit decides between them.
    """
    return sorted(
        native
        for native in RESTORE_NATIVE_FPS
        if native >= effective_unique_fps - MAX_TARGET_FPS_ERROR
        and restore_ratio(source_fps, native) is not None
    )


def restore_native_fps(source_fps: float, effective_unique_fps: float) -> float | None:
    """Slowest isolated-pair native rate that can explain the unique-picture rate."""
    return next(iter(restore_native_fps_candidates(source_fps, effective_unique_fps)), None)


def pair_conversion_native_fps_candidates(row: dict) -> list[float]:
    """Native rates of an isolated-pair conversion described by ``row``, slowest first."""
    groups = row.get("equivalence_groups", [])
    long_share = sum(len(group) > 2 for group in groups) / max(1, len(groups))
    if long_share > RESTORE_MAX_LONG_GROUP_SHARE:
        return []
    return restore_native_fps_candidates(float(row["fps"]), float(row["effective_unique_fps"]))


def pair_conversion_native_fps(row: dict) -> float | None:
    """Slowest native rate of an isolated-pair conversion described by ``row``, else None."""
    return next(iter(pair_conversion_native_fps_candidates(row)), None)


def restore_plan(
    row: dict,
    *,
    first_source_frame: int = 0,
    phase: int | None = None,
    native_fps: float | None = None,
) -> dict:
    """Decide whether a cadence row is a restorable isolated-pair conversion.

    ``row`` is a ``classify_cadence`` dict over consecutive source frames whose first frame is
    absolute 0-based source frame ``first_source_frame``.  With ``phase`` given (a source-wide
    fit), the row only verifies it; otherwise the phase is fitted here and must be unambiguous.
    """
    source_fps = float(row["fps"])
    effective = float(row["effective_unique_fps"])
    count = int(row["frame_count"])
    base = {
        "schema": "cadence_restore_plan_v1",
        "source_fps": source_fps,
        "effective_unique_fps": effective,
        "first_source_frame": first_source_frame,
        "frame_count": count,
    }
    if native_fps is None:
        native_fps = pair_conversion_native_fps(row)
    ratio = None if native_fps is None else restore_ratio(source_fps, native_fps)
    if ratio is None:
        return {**base, "action": "refuse", "reasons": ["not_an_isolated_pair_conversion"]}
    observed = {first_source_frame + int(frame) - 1 for frame in row.get("duplicate_frames", [])}
    fit = fit_restore_phase(
        first=first_source_frame, count=count, observed_repeats=observed, ratio=ratio
    )
    chosen = fit["phase"] if phase is None else int(phase) % ratio[1]
    chosen_score = fit["scores"][chosen]
    reasons = []
    if chosen_score["recall"] < RESTORE_MIN_PREDICTED_RECALL:
        reasons.append("predicted_repeats_not_observed")
    if phase is None and fit["phase_margin"] < RESTORE_MIN_PHASE_MARGIN:
        reasons.append("ambiguous_phase")
    return {
        **base,
        "action": "refuse" if reasons else "restore_native_cadence",
        "reasons": reasons,
        "native_fps": native_fps,
        "ratio_native_to_source": list(ratio),
        "phase": chosen,
        "phase_supplied": phase is not None,
        "predicted_recall": round(chosen_score["recall"], 6),
        "predicted_repeats": chosen_score["predicted"],
        "unexplained_repeats": chosen_score["unexplained_repeats"],
        "phase_margin": fit["phase_margin"],
        # Native capture c is exposed at c/native - phase/source on the source clock.
        "native_time_offset_seconds": -chosen / source_fps,
    }


def restore_timeline(first: int, count: int, plan: dict) -> list[dict]:
    """Map source frames ``[first, first + count)`` (absolute, 0-based) to native captures.

    Each observation names its native capture index, the local 1-based source frames showing
    it, the canonical (first) local frame, and its capture time on the native grid.  A capture
    whose first display precedes ``first`` is kept (its visible member is canonical).
    """
    ratio = tuple(plan["ratio_native_to_source"])
    phase = int(plan["phase"])
    native_fps = float(plan["native_fps"])
    offset = float(plan["native_time_offset_seconds"])
    observations: list[dict] = []
    for frame in range(first, first + count):
        capture = capture_index(frame, phase, ratio)
        local = frame - first + 1
        if observations and observations[-1]["native_capture"] == capture:
            observations[-1]["source_frames"].append(local)
            continue
        observations.append(
            {
                "native_frame": len(observations) + 1,
                "native_capture": capture,
                "canonical_source_frame": local,
                "canonical_absolute_source_frame": frame,
                "source_frames": [local],
                "native_time_seconds": capture / native_fps + offset,
            }
        )
    return observations


def materialize_restored_frames(
    source_frames: Path,
    destination_frames: Path,
    *,
    first: int,
    plan: dict,
    link: str = "hardlink",
) -> list[dict]:
    """Keep one picture per native capture as consecutive ``f_%04d.jpg`` native frames."""
    names = sorted(source_frames.glob("f_*.jpg"))
    timeline = restore_timeline(first, len(names), plan)
    destination_frames.mkdir(parents=True, exist_ok=True)
    for observation in timeline:
        source = source_frames / f"f_{observation['canonical_source_frame']:04d}.jpg"
        destination = destination_frames / f"f_{observation['native_frame']:04d}.jpg"
        if not source.exists():
            raise FileNotFoundError(source)
        if link == "hardlink":
            os.link(source, destination)
        else:
            shutil.copy2(source, destination)
    (destination_frames / "cadence_restore_mapping.json").write_text(
        json.dumps(
            {
                "schema": "cadence_restore_mapping_v1",
                "plan": plan,
                "first_absolute_source_frame": first,
                "source_frames_directory": str(source_frames),
                "observations": timeline,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return timeline


def restore_stream_start(plan: dict) -> dict:
    """First native capture of a whole-source restore whose capture time is not negative.

    Capture ``c`` is exposed at ``c / native + native_time_offset_seconds`` on the source clock;
    the partially shown capture before the source's first frame time is dropped.
    """
    native, source = plan["ratio_native_to_source"]
    phase = int(plan["phase"])
    native_fps = float(plan["native_fps"])
    offset = float(plan["native_time_offset_seconds"])
    capture = capture_index(0, phase, (native, source))
    if capture / native_fps + offset < -1e-9:
        capture += 1
    frame = 0
    while capture_index(frame, phase, (native, source)) < capture:
        frame += 1
    return {
        "first_source_frame": frame,
        "first_native_capture": capture,
        "first_capture_seconds": capture / native_fps + offset,
    }


def restore_select_expression(plan: dict) -> str:
    """FFmpeg ``select`` keeping the first source frame of every native capture.

    ``n`` counts decoded frames from the stream start, so this is only valid on a whole-source
    decode (no seek), which is how ``restore_source_video`` uses it.
    """
    native, source = plan["ratio_native_to_source"]
    phase = int(plan["phase"])
    first = restore_stream_start(plan)["first_source_frame"]
    current = f"floor((n+{phase})*{native}/{source})"
    previous = f"floor((n-1+{phase})*{native}/{source})"
    return f"select='gte(n\\,{first})*(eq(n\\,{first})+not(eq({current}\\,{previous})))'"


def restore_source_video(source: Path, destination: Path, plan: dict, *, crf: int = 14) -> dict:
    """Write the native-cadence picture stream selected by the fitted conversion clock.

    Unlike ``normalize_source_video`` the kept pictures are chosen by frame index, never by
    content, so still pictures survive.  Output frame ``k`` (0-based) is native capture
    ``first_native_capture + k`` stamped at its capture time on the source clock (source start
    at zero), ``first_capture_seconds + k / native_fps``; audio keeps its source times.  One
    re-encode is unavoidable for a whole-stream consumer; the near-transparent CRF keeps it well
    below the source's own compression.  Returns the stream start for provenance.
    """
    if plan.get("action") != "restore_native_cadence":
        raise ValueError("source cadence restore requires a restore plan")
    native_fps = float(plan["native_fps"])
    start = restore_stream_start(plan)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            f"{restore_select_expression(plan)},settb=1/90000,"
            f"setpts=(N/{native_fps}+{start['first_capture_seconds']:.9f})/TB",
            "-fps_mode",
            "passthrough",
            # Without an explicit encoder clock the stamps are rounded onto the source rate.
            "-enc_time_base",
            "1/90000",
            "-video_track_timescale",
            "90000",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            str(crf),
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            "-y",
            str(destination),
        ],
        check=True,
    )
    return start
