"""Probe a source broadcast, then observe its scoreboard at native resolution.

Two artifacts come out of this stage.  ``overlay_crops/`` keeps the historical 960-wide
full-frame observation used by the play-camera clustering.  ``score_crops_native/`` is the
one the reader now uses: the scoreboard overlay is localised once per broadcast from the
temporal-stability map of those proxies, and the resulting box is then cut out of the
1920x1080 source, so the digits reach the model at full sensor resolution instead of at half
scale.  Localisation falls back to the more stable of the top and bottom overlay bands when
no single stable component is found; the fallback is still a native crop and is recorded.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from cv.pipeline import nvdec_video, provenance, source_coverage
from cv.pipeline.cadence_normalize import (
    cadence_action,
    normalize_source_video,
    pair_conversion_native_fps_candidates,
    restore_plan,
    restore_source_video,
)
from cv.pipeline.frame_cadence import SYSTEMATIC_DUPLICATE_RATE, classify_cadence
from cv.pipeline.resolution import FrameSize, meets_native_tracking_minimum, resolution_status
from cv.pipeline.source_timebase import audit_source_timebase


def probe(video: Path) -> dict:
    raw = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,duration:format=duration",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    document = json.loads(raw)
    stream = document["streams"][0]
    duration = stream.get("duration") or document.get("format", {}).get("duration")
    if duration is None:
        raise ValueError(f"ffprobe did not report a duration for {video}")
    fps = float(Fraction(stream["avg_frame_rate"]))
    size = FrameSize(int(stream["width"]), int(stream["height"]))
    return {
        "schema": "broadcast_source_integrity_v1",
        "video": video.name,
        "width": size.width,
        "height": size.height,
        "fps": fps,
        "nominal_fps": float(Fraction(stream["r_frame_rate"])),
        "duration_seconds": float(duration),
        "resolution_valid": meets_native_tracking_minimum(size),
        "resolution_status": resolution_status(size),
    }


def _sample_starts(duration_seconds: float, *, span_seconds: float = 8.0) -> list[float]:
    """Five spread-out samples catch source-wide repeat cadence without a giant probe dump."""
    latest = max(0.0, duration_seconds - span_seconds)
    return sorted({round(latest * fraction, 3) for fraction in (0.0, 0.2, 0.5, 0.8, 1.0)})


def _frame_timestamps(video: Path, start: float, duration: float) -> list[float]:
    raw = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-read_intervals",
            f"{start}%+{duration}",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [
        float(frame["best_effort_timestamp_time"])
        for frame in json.loads(raw).get("frames", [])
        if frame.get("best_effort_timestamp_time") is not None
    ]


def _kept_sample_frames(video: Path, start: float, duration: float, fps: float) -> list[int]:
    """Return local source-frame indices retained by ffmpeg's visual duplicate detector."""
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-vf",
            "mpdecimate=hi=1280:lo=640:frac=0.5,showinfo",
            "-vsync",
            "0",
            "-an",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    from cv.pipeline.frame_cadence import SHOWINFO_TIME

    times = [float(match.group(1)) for match in SHOWINFO_TIME.finditer(completed.stderr)]
    if not times:
        return []
    first = times[0]
    return [round((value - first) * fps) + 1 for value in times]


def source_restore_plan(
    rows: list[dict], timestamp_samples: list[dict], fps: float, combined: dict
) -> dict:
    """Fit one conversion clock across the spread-out samples, or refuse.

    Every candidate native rate is tried.  Exactly one must fit; a source that two rates both
    explain is refused as ambiguous rather than restored on a guess.
    """
    candidates = pair_conversion_native_fps_candidates(combined)
    if not candidates:
        return {"action": "refuse", "reasons": ["not_an_isolated_pair_conversion"], "samples": []}
    plans = {
        native: _restore_plan_at(rows, timestamp_samples, fps, native) for native in candidates
    }
    fitted = [native for native, plan in plans.items() if plan["action"] != "refuse"]
    tried = [
        {"native_fps": native, "action": plan["action"], "reasons": plan.get("reasons", [])}
        for native, plan in plans.items()
    ]
    if len(fitted) > 1:
        return {
            "action": "refuse",
            "reasons": ["ambiguous_native_rate"],
            "samples": [],
            "native_fps_tried": tried,
        }
    chosen = plans[fitted[0] if fitted else candidates[0]]
    return {**chosen, "native_fps_tried": tried}


def _restore_plan_at(
    rows: list[dict], timestamp_samples: list[dict], fps: float, native_fps: float
) -> dict:
    """One candidate native rate: each sample fits its own phase on absolute source frames.

    Samples without motion (every phase explains them) abstain; every decisive sample must
    agree on the phase, so a conversion whose phase breaks inside the broadcast is refused
    rather than half-restored.
    """
    usable_samples = [sample for sample in timestamp_samples if sample["nominal_timeline_valid"]]
    decisive = []
    abstained = []
    for sample, row in zip(usable_samples, rows):
        # The duplicate detector decodes from an accurate seek to ``start_seconds`` (relative to
        # the stream start): its first picture is the first source frame at or after that time.
        first = math.ceil(float(sample["start_seconds"]) * fps - 1e-6)
        plan = restore_plan(row, first_source_frame=first, native_fps=native_fps)
        entry = {
            "clip": row["clip"],
            "first_source_frame": first,
            "action": plan["action"],
            "reasons": plan["reasons"],
            "phase": plan.get("phase"),
            "predicted_recall": plan.get("predicted_recall"),
            "phase_margin": plan.get("phase_margin"),
        }
        if plan["reasons"] == ["ambiguous_phase"]:
            abstained.append(entry)
        else:
            decisive.append((entry, plan))
    samples = [entry for entry, _ in decisive] + abstained
    refusals = sorted({reason for entry, _ in decisive for reason in entry["reasons"]})
    phases = sorted({plan["phase"] for _, plan in decisive})
    if refusals or not decisive or len(phases) != 1:
        return {
            "action": "refuse",
            "reasons": refusals
            or (["phase_changes_between_samples"] if decisive else ["no_decisive_sample"]),
            "samples": samples,
        }
    plan = dict(decisive[0][1])
    plan.update(
        first_source_frame=0,
        frame_count=None,
        samples=samples,
        mapping=(
            "native frame k (1-based) of the restored stream is native capture "
            "c = c(0) + k - 1 with c(j) = floor((j + phase) * ratio[0] / ratio[1]) for "
            "absolute 0-based source frame j; its source frames are {j : c(j) = c} and its "
            "capture time on the source clock is c / native_fps + native_time_offset_seconds"
        ),
    )
    return plan


# A point-gated source may have at most this share of frames with a local timestamp defect
# (gap, repeat or missing PTS); more than that is a broken clock, not a local defect.
POINT_GATED_MAX_LOCAL_DEFECT_SHARE = 0.01
#: last decoded picture this close to the container duration counts as decoded to the end
DECODED_TO_END_TOLERANCE_SECONDS = 1.0


def _sample_regularity(timestamps: list[float], fps: float) -> dict:
    pts_tolerance = max(0.0011, 0.02 / fps)
    increasing = (
        len(timestamps) >= 2
        and all(math.isfinite(value) for value in timestamps)
        and all(right > left for left, right in zip(timestamps, timestamps[1:]))
    )
    intervals = [right - left for left, right in zip(timestamps, timestamps[1:])]
    maximum_phase_error = (
        max(abs(value - timestamps[0] - index / fps) for index, value in enumerate(timestamps))
        if increasing
        else None
    )
    regular = bool(
        increasing
        and maximum_phase_error <= pts_tolerance
        and all(abs(interval - 1.0 / fps) <= pts_tolerance for interval in intervals)
    )
    return {
        "frames": len(timestamps),
        "first_pts_seconds": timestamps[0] if timestamps else None,
        "last_pts_seconds": timestamps[-1] if timestamps else None,
        "strictly_increasing": increasing,
        "nominal_timeline_valid": regular,
        "maximum_nominal_phase_error_seconds": maximum_phase_error,
        "pts_tolerance_seconds": pts_tolerance,
        "min_interval_seconds": min(intervals, default=0.0),
        "max_interval_seconds": max(intervals, default=0.0),
    }


def _decoded_to_end(full_timebase: dict, duration: float) -> bool:
    """Decoder complained (e.g. H.264 reference errors at a broadcast splice) yet exited cleanly
    and its pictures reach the container's end: the errors are local, not a truncated tail.

    Buse-Humbert (pilot 2026-09-29): two ~6 s splices, 719,994 of 720,720 frames, last PTS at
    the container duration; refused whole before this. The spans stay point-gated.
    """
    last = full_timebase.get("last_pts_seconds")
    return (
        full_timebase.get("decoder_return_code") == 0
        and isinstance(last, (int, float))
        and math.isfinite(last)
        and last >= duration - DECODED_TO_END_TOLERANCE_SECONDS
    )


def _point_gated(reasons: list[str], fps: float, **fields) -> dict:
    """Process the native pictures and let the per-point gates hold what the defect touches.

    The point cadence gate re-audits every extracted point for repeated pictures and for
    timestamp gaps in its own extraction clock, so a local source defect holds only the
    points it reaches instead of the whole broadcast.
    """
    return {
        "schema": "source_cadence_audit_v1",
        "decision": "point_gated",
        "action": "retain_native_point_gated",
        "reason": ";".join(reasons),
        "processing_fps": fps,
        **fields,
    }


def source_cadence(
    video: Path,
    metadata: dict,
    *,
    restore: bool = False,
    point_gate: bool = True,
    full_scan: bool = True,
    inventory: Path | None = None,
) -> dict:
    """Audit source PTS and repeated pictures before any point-local extraction.

    With ``restore`` a systematic isolated-pair conversion (e.g. 25 fps doubled to 50, or
    pulled down to 30) is restored to its native cadence by the fitted conversion clock
    instead of the content-decimating heal; a pair conversion that does not fit stays refused.

    With ``point_gate`` (the product default) a source that can be neither used as is nor
    healed is not refused whole: it is processed at its native cadence as ``point_gated`` and
    the per-point cadence gate holds only the points a defect reaches. It is still refused
    when the decoder fails or when local timestamp defects exceed
    ``POINT_GATED_MAX_LOCAL_DEFECT_SHARE`` of all frames. A container whose average rate
    disagrees with a nominal rate its samples follow exactly is audited on the nominal clock
    (``metadata["fps"]`` becomes the nominal rate; the average is kept).

    ``full_scan=False`` skips the every-frame PTS scan (tens of minutes on a full match):
    the verdict then predicts the pipeline's from the samples alone, for a quick
    check (``cadence_precheck``). ``inventory`` keeps the full scan's every-frame PTS lines.
    """
    fps = float(metadata["fps"])
    duration = float(metadata["duration_seconds"])
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError("source fps and duration must be finite and positive")
    # Millisecond-quantized containers are representable on the nominal timeline;
    # genuine VFR/gaps/drift are not. Check accumulated phase as well as local intervals.
    span = min(8.0, duration)
    starts = _sample_starts(duration, span_seconds=span)
    sampled = [_frame_timestamps(video, start, span) for start in starts]
    checks = [_sample_regularity(timestamps, fps) for timestamps in sampled]
    nominal = float(metadata.get("nominal_fps") or fps)
    if (
        point_gate
        and not all(check["nominal_timeline_valid"] for check in checks)
        and math.isfinite(nominal)
        and nominal > 0
        and not math.isclose(nominal, fps, rel_tol=1e-6)
    ):
        nominal_checks = [_sample_regularity(timestamps, nominal) for timestamps in sampled]
        if all(check["nominal_timeline_valid"] for check in nominal_checks):
            # e.g. 59.88 average over a 59.94 clock: frames are missing somewhere, and the
            # full scan below locates those gaps on the true clock.
            metadata["container_average_fps"] = fps
            metadata["fps"] = fps = nominal
            checks = nominal_checks
    rows = []
    timestamp_samples = []
    for index, (start, check) in enumerate(zip(starts, checks)):
        timestamp_samples.append({"start_seconds": start, **check})
        if not check["nominal_timeline_valid"]:
            continue
        kept = _kept_sample_frames(video, start, span, fps)
        rows.append(
            classify_cadence(
                match_id="source",
                clip=f"sample{index:02d}",
                fps=fps,
                frame_count=check["frames"],
                kept_frames=kept,
                active_spans=[[1.0, float(check["frames"])]],
            ).as_dict()
        )
    samples_regular = bool(rows) and all(row["nominal_timeline_valid"] for row in timestamp_samples)
    if not samples_regular and not point_gate:
        return {
            "schema": "source_cadence_audit_v1",
            "decision": "rejected",
            "action": "discard_invalid_timebase",
            "reason": "irregular_source_timestamps",
            "timestamp_samples": timestamp_samples,
        }
    full_timebase = (
        audit_source_timebase(video, fps, inventory)
        if full_scan
        else {"nominal_timeline_valid": True, "scope": "not_scanned_samples_only"}
    )
    timebase_valid = full_timebase.get("nominal_timeline_valid") is True
    gated_reasons: list[str] = []
    if not timebase_valid:
        local_share = int(full_timebase.get("local_defect_frames") or 0) / max(
            1, int(full_timebase.get("frames") or 0)
        )
        refused = (
            "invalid_full_source_timebase"
            if not point_gate
            else (
                "source_decode_incomplete"
                if full_timebase.get("completed") is False
                and not _decoded_to_end(full_timebase, duration)
                else (
                    "widespread_timebase_defects"
                    if local_share > POINT_GATED_MAX_LOCAL_DEFECT_SHARE
                    or full_timebase.get("local_defect_spans_truncated")
                    else None
                )
            )
        )
        if refused:
            return {
                "schema": "source_cadence_audit_v1",
                "decision": "rejected",
                "action": "discard_invalid_timebase",
                "reason": refused,
                "timestamp_samples": timestamp_samples,
                "full_timebase": full_timebase,
            }
        gated_reasons.append(
            "local_timebase_defects" if samples_regular else "irregular_source_timestamps"
        )
        if full_timebase.get("completed") is False:
            gated_reasons.append("decoder_errors_decoded_to_end")
    if not rows:
        return _point_gated(
            [*gated_reasons, "no_regular_sample_for_repeat_audit"],
            fps,
            timestamp_samples=timestamp_samples,
            full_timebase=full_timebase,
        )
    # Join equally-sized local samples into one cadence observation.  Frame numbers only
    # identify repeat pattern here; PTS remains recorded per sample above.
    offset = 0
    duplicate_frames: list[int] = []
    groups: list[list[int]] = []
    for row in rows:
        duplicate_frames.extend(offset + int(frame) for frame in row["duplicate_frames"])
        groups.extend(
            [[offset + int(frame) for frame in group] for group in row["equivalence_groups"]]
        )
        offset += int(row["frame_count"])
    kept = sorted(set(range(1, offset + 1)) - set(duplicate_frames))
    combined = classify_cadence(
        match_id="source",
        clip="source_samples",
        fps=fps,
        frame_count=offset,
        kept_frames=kept,
        active_spans=[[1.0, float(offset)]],
    ).as_dict()
    # ``classify_cadence`` calls a sample set with any hold longer than five frames usable (a
    # still picture is not a conversion), so one still in a doubled broadcast hides its
    # systematic repeats.  The restore fit does not need that guard: stills abstain per sample.
    if restore and (
        combined["decision"] == "timing_hold"
        or combined["active_duplicate_rate"] >= SYSTEMATIC_DUPLICATE_RATE
    ):
        plan = source_restore_plan(rows, timestamp_samples, fps, combined)
        pair_conversion = "not_an_isolated_pair_conversion" not in plan["reasons"]
        if plan["action"] == "restore_native_cadence" or pair_conversion:
            restored = plan["action"] == "restore_native_cadence"
            fields = {
                "effective_unique_fps": combined["effective_unique_fps"],
                "duplicate_pattern": combined,
                "cadence_restore": plan,
                "timestamp_samples": timestamp_samples,
                "full_timebase": full_timebase,
            }
            if point_gate and (gated_reasons or not restored):
                # A restore cannot map pictures across timestamp gaps, and a refused restore
                # plan leaves the repeats to the point gate.
                why = [] if restored else [f"restore_refused:{';'.join(plan['reasons'])}"]
                return _point_gated([*gated_reasons, *why], fps, **fields)
            return {
                "schema": "source_cadence_audit_v1",
                "decision": "restored" if restored else "rejected",
                "action": "restore_native_cadence" if restored else "discard_invalid_timebase",
                "reason": ";".join(plan["reasons"]),
                "processing_fps": plan.get("native_fps") if restored else None,
                **fields,
            }
    action = cadence_action(combined)
    fields = {
        "effective_unique_fps": combined["effective_unique_fps"],
        "duplicate_pattern": combined,
        "cadence_action": action,
        "timestamp_samples": timestamp_samples,
        "full_timebase": full_timebase,
    }
    if point_gate and (gated_reasons or action["action"] == "discard_invalid_timebase"):
        # The content heal re-times the whole stream, which a timestamp gap would silently
        # shift; irregular repeats are left to the point gate.
        why = [f"heal_refused:{reason}" for reason in action["reasons"]]
        return _point_gated([*gated_reasons, *why], fps, **fields)
    final_decision = (
        "usable"
        if action["action"] == "retain_native"
        else ("healed" if action["action"] == "heal_regular_duplicate_cadence" else "rejected")
    )
    return {
        "schema": "source_cadence_audit_v1",
        "decision": final_decision,
        "action": action["action"],
        "reason": ";".join(action["reasons"]),
        "processing_fps": action["target_fps"],
        **fields,
    }


def cadence_precheck(video: Path) -> dict:
    """Quick verdict on a new file: would the product use, heal, restore, point-gate or refuse it?

    Samples only (a few seconds of decode); the production upstream repeats the full audit.
    """
    metadata = probe(video)
    cadence = source_cadence(video, metadata, restore=True, full_scan=False)
    return {
        "schema": "source_cadence_precheck_v1",
        "decision": cadence["decision"],
        "reason": cadence.get("reason", ""),
        "fps": metadata["fps"],
        "container_average_fps": metadata.get("container_average_fps"),
        "processing_fps": cadence.get("processing_fps"),
        "effective_unique_fps": cadence.get("effective_unique_fps"),
        "full_timebase": "not_scanned",
    }


#: The full timebase scan's every-frame PTS (FFprobe compact lines) and the receipt binding
#: them to the source bytes; the S6 source clock reuses them instead of a second full decode.
SOURCE_PTS_INVENTORY = "source_pts_inventory_v1.txt"
SOURCE_PTS_INVENTORY_RECEIPT = "source_pts_inventory_v1.json"


def write_pts_inventory_receipt(video: Path, cadence: dict, inventory: Path) -> Path:
    full = cadence.get("full_timebase") or {}
    receipt = inventory.with_name(SOURCE_PTS_INVENTORY_RECEIPT)
    receipt.write_text(
        json.dumps(
            {
                "schema": "source_pts_inventory_v1",
                "video": provenance.file_record(video, role="processing_video"),
                "command": full.get("command"),
                "completed": full.get("completed"),
                "output": provenance.file_record(inventory),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return receipt


def processing_source(video: Path, out: Path, integrity: dict) -> tuple[Path, float]:
    """Resolve the audited picture stream and its clock as one inseparable input."""
    cadence = integrity["cadence"]
    if cadence.get("decision") not in {"usable", "healed", "restored", "point_gated"}:
        raise ValueError("source cadence is not usable")
    fps = float(cadence["processing_fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("invalid processing cadence")
    normalized = cadence.get("normalized_video")
    if cadence["decision"] in {"healed", "restored"}:
        if not isinstance(normalized, str) or not normalized or Path(normalized).name != normalized:
            raise ValueError("healed or restored cadence requires a named normalized video")
        selected = out / normalized
        if not selected.is_file():
            raise FileNotFoundError(f"audited normalized video is missing: {selected}")
        return selected, fps
    if normalized or not math.isclose(fps, float(integrity["fps"]), rel_tol=1e-6):
        raise ValueError("native source and processing cadence disagree")
    if not video.is_file():
        raise FileNotFoundError(video)
    return video, fps


SCOREBOARD_SAMPLES = 360
SCOREBOARD_PAD_X = 0.12
SCOREBOARD_PAD_Y = 0.5
# The stable component is the part of the overlay that does not change: the names and the
# completed sets.  The current games and the points box change every point, so they are not in
# it, and they sit on the side the board grows towards - away from the frame edge it hugs.
SCOREBOARD_PAD_GROWTH = 0.75


def _stability_maps(paths: list[Path]) -> list[tuple[np.ndarray, np.ndarray]]:
    """(agreement, edge) map pairs over all sampled frames and over the modal scene only.

    The overlay is only drawn on some shots, so the all-frames map can be washed out on
    broadcasts that cut away often.  Restricting to the frames nearest the median picture
    keeps the play camera, where the overlay is almost always present.
    """
    step = max(1, len(paths) // SCOREBOARD_SAMPLES)
    sample = paths[::step][:SCOREBOARD_SAMPLES]
    stack = np.stack([cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in sample]).astype(
        np.float32
    )
    maps = []
    for modal in (False, True):
        frames = stack
        if modal:
            distance = np.abs(stack - np.median(stack, axis=0)).mean(axis=(1, 2))
            keep = np.argsort(distance)[: max(20, int(0.45 * len(stack)))]
            frames = stack[keep]
        median = np.median(frames, axis=0)
        agreement = (np.abs(frames - median) < 8).mean(axis=0)
        edge = np.abs(cv2.Laplacian(cv2.GaussianBlur(median, (3, 3), 0), cv2.CV_32F, ksize=3))
        maps.append((agreement, edge))
    return maps


def _components(agreement: np.ndarray, edge: np.ndarray) -> list[tuple[float, int, int, int, int]]:
    height, width = agreement.shape
    agreement_threshold = min(0.85, max(0.30, float(np.percentile(agreement, 97))))
    edge_threshold = min(30.0, max(10.0, float(np.percentile(edge, 93))))
    mask = ((agreement > agreement_threshold) & (edge > edge_threshold)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (51, 13))
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    found = []
    for index in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[index])
        if box_width < 0.06 * width or box_height < 0.015 * height:
            continue
        if box_width < 1.8 * box_height or box_width > 0.8 * width or box_height > 0.35 * height:
            continue
        centre = (y + box_height / 2) / height
        if 0.22 < centre < 0.68:  # the overlay lives in the top or bottom band
            continue
        score = area * float(agreement[y : y + box_height, x : x + box_width].mean())
        found.append((score, x, y, box_width, box_height))
    return found


def _overlap(first: tuple, second: tuple) -> float:
    """Intersection over union of two (score, x, y, width, height) components."""
    _, x, y, w, h = first
    _, u, v, p, q = second
    inter = max(0, min(x + w, u + p) - max(x, u)) * max(0, min(y + h, v + q) - max(y, v))
    return inter / float(w * h + p * q - inter)


def locate_scoreboard(paths: list[Path], *, map_agreement: bool = False) -> dict:
    """Normalised scoreboard box for this broadcast, from the overlay stability map.

    ``map_agreement`` (optional, default off): prefer a component that both stability maps
    find. A fixed single-camera feed is stable everywhere, so its static sponsor boards can
    outscore the board in the all-frames map; the overlay itself is found by both maps
    (Mladenovic-Zhang 2018: the boards won and every score read was empty). On the twenty
    dry-run broadcasts the choice changed only there (cv/experiments/long_shot_view).
    """
    maps = _stability_maps(paths)
    height, width = maps[0][0].shape
    found = [_components(agreement, edge) for agreement, edge in maps]
    candidates: list[tuple[float, int, int, int, int]] = [*found[0], *found[1]]
    agreed = [
        component
        for own, other in ((found[0], found[1]), (found[1], found[0]))
        for component in own
        if any(_overlap(component, match) > 0.5 for match in other)
    ]
    if map_agreement and agreed:
        _, x, y, box_width, box_height = max(agreed)
        method = "stability_component_map_agreement"
    elif candidates:
        _, x, y, box_width, box_height = max(candidates)
        method = "stability_component"
    else:
        agreement, edge = maps[1]
        strength = agreement * np.minimum(edge, 60.0)
        band = max(1, int(0.22 * height))
        top = float(strength[:band, : int(0.6 * width)].mean())
        bottom = float(strength[height - band :, : int(0.6 * width)].mean())
        y = 0 if top >= bottom else height - band
        x, box_width, box_height = 0, int(0.5 * width), band
        method = "band_fallback_top" if top >= bottom else "band_fallback_bottom"
    pad_x = max(10, int(SCOREBOARD_PAD_X * box_width))
    pad_y = max(10, int(SCOREBOARD_PAD_Y * box_height))
    growth = max(pad_x, int(SCOREBOARD_PAD_GROWTH * box_width))
    grows_right = (x + box_width / 2) < width / 2
    left = max(0, x - (pad_x if grows_right else growth))
    right = min(width, x + box_width + (growth if grows_right else pad_x))
    top_edge = max(0, y - pad_y)
    bottom_edge = min(height, y + box_height + pad_y)
    return {
        "method": method,
        "grows_right": bool(grows_right),
        "proxy_size": [width, height],
        "box_normalised": [
            left / width,
            top_edge / height,
            right / width,
            bottom_edge / height,
        ],
    }


SCOREBOARD_REFINE_PAD = 0.06
# The points box is the least stable part of the board - it changes every point - so the
# stability map stops just short of it.  The board grows towards the points box, so the
# refined box is extended that way by a fifth of its width.
SCOREBOARD_REFINE_GROWTH = 0.22


def refine_scoreboard_box(
    paths: list[Path], *, samples: int = 192, grows_right: bool = True
) -> dict | None:
    """A tighter box around the overlay inside an already-cut scoreboard crop.

    The first cut is deliberately generous - it is padded by three quarters of its width
    towards the side the board grows - so the board can occupy less than half of it and the
    reader is shown a small board in a large picture.  The overlay is the part of the crop
    that does not change from second to second, so the same stability argument that found
    the board in the frame finds its edges inside the crop, and the result can be cut and
    enlarged without decoding the video again.
    """
    step = max(1, len(paths) // samples)
    frames = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in paths[::step][:samples]]
    frames = [frame for frame in frames if frame is not None]
    if len(frames) < 8:
        return None
    stack = np.stack(frames).astype(np.float32)
    height, width = stack.shape[1:]
    median = np.median(stack, axis=0)
    agreement = (np.abs(stack - median) < 8).mean(axis=0)
    edge = np.abs(cv2.Laplacian(cv2.GaussianBlur(median, (3, 3), 0), cv2.CV_32F, ksize=3))
    # The overlay is only drawn on some shots, so "stable" is relative to this crop rather
    # than an absolute share of frames.
    agreement_threshold = min(0.80, max(0.30, float(np.percentile(agreement, 85))))
    mask = (
        (agreement > agreement_threshold) & (edge > max(10.0, float(np.percentile(edge, 90))))
    ).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (41, 21))
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for index in range(1, count):
        x, y, box_width, box_height, area = (int(value) for value in stats[index])
        if box_width < 0.15 * width or box_height < 0.10 * height:
            continue
        if best is None or area > best[0]:
            best = (area, x, y, box_width, box_height)
    if best is None:
        return None
    _, x, y, box_width, box_height = best
    pad_x = max(2, int(SCOREBOARD_REFINE_PAD * box_width))
    pad_y = max(2, int(SCOREBOARD_REFINE_PAD * box_height))
    growth = max(pad_x, int(SCOREBOARD_REFINE_GROWTH * box_width))
    left = max(0, x - (pad_x if grows_right else growth))
    top = max(0, y - pad_y)
    right = min(width, x + box_width + (growth if grows_right else pad_x))
    bottom = min(height, y + box_height + pad_y)
    return {
        "crop_size": [int(width), int(height)],
        "box_normalised": [left / width, top / height, right / width, bottom / height],
        "box_pixels_xywh": [left, top, right - left, bottom - top],
    }


def tighten_scoreboard_crops(
    paths: list[Path], out_dir: Path, box: list[float], *, scale: float = 2.0
) -> int:
    """Re-cut and enlarge existing scoreboard crops; no video decoding."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        left = max(0, int(round(box[0] * width)))
        top = max(0, int(round(box[1] * height)))
        right = max(left + 8, min(width, int(round(box[2] * width))))
        bottom = max(top + 8, min(height, int(round(box[3] * height))))
        cut = image[top:bottom, left:right]
        if scale != 1.0:
            cut = cv2.resize(cut, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(out_dir / path.name), cut, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        written += 1
    return written


def extract_native_scoreboard_crops(
    video: Path,
    out_dir: Path,
    box: list[float],
    *,
    source_width: int,
    source_height: int,
    fps: float,
) -> tuple[int, list[int]]:
    """Cut the located box out of the 1920x1080 source, one crop per observation second."""
    left = int(round(box[0] * source_width)) & ~1
    top = int(round(box[1] * source_height)) & ~1
    right = min(source_width, int(round(box[2] * source_width)))
    bottom = min(source_height, int(round(box[3] * source_height)))
    crop_width = max(16, (right - left) & ~1)
    crop_height = max(16, (bottom - top) & ~1)
    out_dir.mkdir(parents=True, exist_ok=True)
    nvdec_video.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            f"fps={fps},crop={crop_width}:{crop_height}:{left}:{top}",
            "-q:v",
            "2",
            str(out_dir / "s_%06d.jpg"),
        ],
        check=True,
    )
    return len(list(out_dir.glob("s_*.jpg"))), [left, top, crop_width, crop_height]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--score-fps", type=float, default=1.0)
    parser.add_argument(
        "--cadence-restore",
        action="store_true",
        help="restore isolated-pair repeat conversions (25->50, 25->30) to native cadence",
    )
    parser.add_argument(
        "--scoreboard-map-agreement",
        action="store_true",
        help="prefer a scoreboard component both stability maps find (fixed-camera feeds)",
    )
    parser.add_argument(
        "--refuse-source-defects",
        action="store_true",
        help=(
            "rollback: refuse the whole broadcast on irregular repeats or a local timestamp "
            "defect instead of processing it point-gated"
        ),
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = probe(args.video)
    if not metadata["resolution_valid"]:
        raise ValueError(
            f"source is {metadata['width']}x{metadata['height']}; canonical minimum is 1920x1080"
        )
    inventory = args.out / SOURCE_PTS_INVENTORY
    inventory.unlink(missing_ok=True)
    metadata["cadence"] = source_cadence(
        args.video,
        metadata,
        restore=args.cadence_restore,
        point_gate=not args.refuse_source_defects,
        inventory=inventory,
    )
    if inventory.is_file():
        write_pts_inventory_receipt(args.video, metadata["cadence"], inventory)
    integrity = args.out / "source_integrity.json"
    integrity.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    if metadata["cadence"]["decision"] == "rejected":
        raise ValueError(f"source cadence rejected: {metadata['cadence']['reason']}")
    if metadata["cadence"]["action"] == "heal_regular_duplicate_cadence":
        normalized = args.out / "cadence_normalized_source.mp4"
        normalize_source_video(
            args.video,
            normalized,
            float(metadata["cadence"]["processing_fps"]),
        )
        metadata["cadence"]["normalized_video"] = normalized.name
        integrity.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    if metadata["cadence"]["action"] == "restore_native_cadence":
        restored = args.out / "cadence_restored_source.mp4"
        metadata["cadence"]["cadence_restore"]["restored_stream"] = restore_source_video(
            args.video, restored, metadata["cadence"]["cadence_restore"]
        )
        metadata["cadence"]["normalized_video"] = restored.name
        integrity.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    frames = args.out / "overlay_crops"
    frames.mkdir(parents=True, exist_ok=True)
    nvdec_video.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(args.video),
            "-vf",
            f"fps={args.score_fps},scale=960:-2",
            "-q:v",
            "4",
            str(frames / "s_%06d.jpg"),
        ],
        check=True,
    )
    crop_paths = sorted(frames.glob("s_*.jpg"))
    if not crop_paths:
        raise ValueError(f"no scoreboard observation frames written to {frames}")
    duration = float(metadata["duration_seconds"])
    overlay_coverage = source_coverage.check(
        "overlay_crops", len(crop_paths) / args.score_fps, duration
    )
    location = locate_scoreboard(crop_paths, map_agreement=args.scoreboard_map_agreement)
    native_dir = args.out / "score_crops_native"
    written, pixel_box = extract_native_scoreboard_crops(
        args.video,
        native_dir,
        location["box_normalised"],
        source_width=metadata["width"],
        source_height=metadata["height"],
        fps=args.score_fps,
    )
    (args.out / "score_crops_native.coordinates.json").write_text(
        json.dumps(
            {
                "schema": "native_scoreboard_crops_v1",
                "fps": args.score_fps,
                "source_width": metadata["width"],
                "source_height": metadata["height"],
                "crop_pixels_xywh": pixel_box,
                "localisation": location,
                "frames": written,
                "resolution_status": "NATIVE",
                **(
                    {"coverage": coverage}
                    if (
                        coverage := source_coverage.check(
                            "score_crops_native", written / args.score_fps, duration
                        )
                    )
                    else {}
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (args.out / "overlay_crops.coordinates.json").write_text(
        json.dumps(
            {
                "schema": "scoreboard_observation_frames_v1",
                "fps": args.score_fps,
                "source_width": metadata["width"],
                "source_height": metadata["height"],
                "artifact_width": 960,
                "artifact_height": round(960 * metadata["height"] / metadata["width"]),
                "crop": "full_frame",
                **({"coverage": overlay_coverage} if overlay_coverage else {}),
                "resolution_status": "LEGACY_BAD_SHOULD_UPDATE",
                "subnative_justification": "scoreboard-only observation proxy; never tracking evidence",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
