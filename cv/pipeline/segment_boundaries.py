"""Move coarse S1 attempt boundaries onto the owner's physical point definition.

The S1 serve/end heads deliberately keep a tail so a rally is never cut short, so
``rally_t_end`` marks a structural boundary (the next start, the shot end, the score
change, or the last audio onset plus a buffer), not the physical ending.  Measured on the
110 SEG-001 source attempts that tail is a systematic +40 frame bias.  ``rally_t_start``
is likewise the serve-head peak, not the toss release.

This stage re-places both boundaries from label-free post-segmentation evidence:

* ``player_tracks_native_v1.csv`` - both players in court metres on source time.  A rally
  keeps at least one player above a running speed; the physical ending is where that stops.
* ``serve_audio_peaks_v1.csv`` - the full-match racket-impact train.  An impact peak sits
  within a few frames of the physical ending in nearly every attempt, so the motion estimate
  is snapped to the nearest peak; the peak train alone cannot pick which peak that is.
* the service motion duration, a calibrated constant, places the toss release before the
  first contact the ledger already carries.

Every witness abstains rather than inventing a boundary, and an abstention preserves the
coarse S1 value so no attempt is lost.  Nothing here reads labels or reviewed spans.

The rally-speed threshold and the service-motion duration below are calibrated on the four
SEG-001 development broadcasts.  Chosen leave-one-broadcast-out they give held-out medians of
12.50 frames (ending) and 6.25 (toss) against 11.50 and 4.50 with the shipped values, so the
calibration is not carrying the measurement.  See ``REPORT.md`` for the full comparison and
for the rejected candidates (audio-only selection, the camera cut, the ball track).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cv.pipeline import provenance

LEDGER_NAME = "automatic_point_ledger_v1.csv"
PEAKS_NAME = "serve_audio_peaks_v1.csv"
TRACKS_NAME = "player_tracks_native_v1.csv"
REPORT_NAME = "boundary_refinement_report_v1.json"

# Court metres: the near baseline is y=0 and the far baseline y=23.77.
BASELINE_COURT_Y = {"near": 0.0, "far": 23.77}

SPEED_SMOOTHING_SECONDS = 0.40
RALLY_SPEED_METRES_PER_SECOND = 2.25
ENDING_SNAP_FRAMES = 16.0
SHORT_ATTEMPT_WINDOW_SECONDS = 1.50
SHORT_ATTEMPT_MINIMUM_SAMPLES = 5
ENDING_LEAD_IN_SECONDS = 0.30
SERVE_MOTION_SECONDS = 1.30
SERVE_STANCE_HOLD_SECONDS = 1.20
SERVE_STANCE_SPEED = 1.10
SERVE_STANCE_DEPTH_METRES = 0.20
SECOND_SERVE_MINIMUM_SEPARATION_SECONDS = 7.0
SECOND_SERVE_IMPACT_WINDOW_FRAMES = (-4.0, 25.0)
FALSE_SPLIT_MAXIMUM_GAP_SECONDS = 4.0


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# ---------------------------------------------------------------- evidence loaders


def load_audio_peaks(path: Path) -> np.ndarray:
    """Return the ``(source_seconds, robust_z)`` impact train, sorted by time."""
    rows = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            time, strength = _finite(row.get("t")), _finite(row.get("z"))
            if time is not None and strength is not None:
                rows.append((time, strength))
    return np.asarray(sorted(rows), dtype=float).reshape(-1, 2)


def load_player_motion(
    path: Path,
    *,
    fps: float,
    clip: str | None = None,
    native_pts_by_frame: Mapping[int, float] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per side: source times, court metres, and smoothed court speed.

    ``player_tracks_native_v1.csv`` carries the source timestamp of every associated
    detection; gaps stay gaps because no sample is synthesised across them.
    """
    if (clip is None) != (native_pts_by_frame is None):
        raise ValueError("explicit clip and native PTS mapping must be supplied together")
    if native_pts_by_frame is not None:
        if not native_pts_by_frame or any(
            type(frame) is not int or frame < 1 or not math.isfinite(float(t))
            for frame, t in native_pts_by_frame.items()
        ):
            raise ValueError("finite point-local native PTS mapping required")
        ordered = [native_pts_by_frame[f] for f in sorted(native_pts_by_frame)]
        if any(b <= a for a, b in zip(ordered, ordered[1:])):
            raise ValueError("strictly increasing original native PTS required")
    samples: dict[str, dict[float, tuple[float, float]]] = defaultdict(dict)
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if clip is not None and row.get("clip") != clip:
                continue
            time = _finite(row.get("t"))
            if native_pts_by_frame is not None:
                from cv.experiments.connected_shooting.auto_packet import native_frame

                frame = native_frame(row.get("frame"))
                if frame not in native_pts_by_frame:
                    continue
                expected = float(native_pts_by_frame[frame])
                if row.get("t") not in (None, ""):
                    # Existing motion storage rounds source seconds to milliseconds.
                    if time is None or abs(time - expected) > 0.000501:
                        raise ValueError("player timestamp differs from declared native source PTS")
                else:
                    time = expected
            x, y = _finite(row.get("court_x")), _finite(row.get("court_y"))
            side = str(row.get("side") or "")
            if time is None or x is None or y is None or side not in BASELINE_COURT_Y:
                continue
            samples[side][round(time, 3)] = (x, y)
    motion = {}
    for side, mapping in samples.items():
        times = np.asarray(sorted(mapping), dtype=float)
        if len(times) < 5:
            continue
        court = np.asarray([mapping[time] for time in times], dtype=float)
        step = np.maximum(np.gradient(times), 1e-3)
        speed = np.hypot(np.gradient(court[:, 0]), np.gradient(court[:, 1])) / step
        window = max(1, int(round(SPEED_SMOOTHING_SECONDS * fps)))
        motion[side] = (times, court, np.convolve(speed, np.ones(window) / window, mode="same"))
    return motion


# ---------------------------------------------------------------- witnesses


def physical_ending_witness(
    motion: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    peaks: np.ndarray,
    *,
    first_impact_seconds: float,
    coarse_end_seconds: float,
    fps: float,
) -> dict[str, Any]:
    """Place the physical ending where rally running stops, snapped to an impact peak.

    The witness is a segmentation boundary, not a certified terminal event: it does not
    claim to have seen the second bounce, only that the players stopped playing the ball.
    """
    rally_end = None
    for side, (times, _court, speed) in motion.items():
        inside = (
            (times >= first_impact_seconds + ENDING_LEAD_IN_SECONDS)
            & (times <= coarse_end_seconds)
            & (speed >= RALLY_SPEED_METRES_PER_SECOND)
        )
        if inside.any():
            latest = float(times[inside][-1])
            rally_end = latest if rally_end is None else max(rally_end, latest)
    if rally_end is None:
        # No player ever reached rally speed.  With the players observed, that is a short
        # attempt (ace, fault, or a ball nobody chased), and the last impact shortly after
        # the serve contact is the terminal one.  With nobody observed there is no witness.
        observed = motion_samples(
            motion, start_seconds=first_impact_seconds, end_seconds=coarse_end_seconds
        )
        window = (
            peaks[
                (peaks[:, 0] >= first_impact_seconds + ENDING_LEAD_IN_SECONDS)
                & (peaks[:, 0] <= first_impact_seconds + SHORT_ATTEMPT_WINDOW_SECONDS)
                & (peaks[:, 0] <= coarse_end_seconds)
            ]
            if len(peaks)
            else np.empty((0, 2))
        )
        if observed < SHORT_ATTEMPT_MINIMUM_SAMPLES or not len(window):
            return {
                "schema": "s1_physical_ending_witness_v1",
                "frame": None,
                "seconds": None,
                "confidence": 0.0,
                "abstained": True,
                "abstention_reason": "no_player_motion_evidence_in_segment"
                if observed < SHORT_ATTEMPT_MINIMUM_SAMPLES
                else "no_impact_after_short_attempt_contact",
                "snapped_to_impact": False,
                "witness": None,
            }
        return {
            "schema": "s1_physical_ending_witness_v1",
            "frame": float(window[-1, 0]) * fps,
            "seconds": float(window[-1, 0]),
            "confidence": 0.25,
            "abstained": False,
            "abstention_reason": None,
            "snapped_to_impact": True,
            "witness": "short_attempt_impact_train",
        }
    seconds, snapped = rally_end, False
    if len(peaks):
        nearby = peaks[np.abs(peaks[:, 0] - rally_end) * fps <= ENDING_SNAP_FRAMES]
        if len(nearby):
            seconds = float(nearby[np.argmin(np.abs(nearby[:, 0] - rally_end)), 0])
            snapped = True
    return {
        "schema": "s1_physical_ending_witness_v1",
        "frame": seconds * fps,
        "seconds": seconds,
        "motion_seconds": rally_end,
        "confidence": 0.6 if snapped else 0.4,
        "abstained": False,
        "abstention_reason": None,
        "snapped_to_impact": snapped,
        "witness": "rally_player_motion",
    }


def toss_release_witness(*, first_impact_seconds: float | None, fps: float) -> dict[str, Any]:
    """Place the toss release one service motion before the ledger's first contact.

    The service motion duration is a calibrated constant, not a detected release; the
    witness therefore inherits every error in the first-contact estimate it is anchored to.
    """
    if first_impact_seconds is None:
        return {
            "schema": "s1_toss_release_witness_v1",
            "frame": None,
            "seconds": None,
            "confidence": 0.0,
            "abstained": True,
            "abstention_reason": "no_automatic_first_contact",
        }
    seconds = first_impact_seconds - SERVE_MOTION_SECONDS
    return {
        "schema": "s1_toss_release_witness_v1",
        "frame": seconds * fps,
        "seconds": seconds,
        "confidence": 0.5,
        "abstained": False,
        "abstention_reason": None,
        "service_motion_seconds": SERVE_MOTION_SECONDS,
    }


def motion_samples(
    motion: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    start_seconds: float,
    end_seconds: float,
) -> int:
    """Associated player samples covering a span; zero means no evidence at all."""
    return max(
        (
            int(((times >= start_seconds) & (times <= end_seconds)).sum())
            for times, _, _ in motion.values()
        ),
        default=0,
    )


def serve_stance_marks(
    motion: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    start_seconds: float,
    end_seconds: float,
) -> list[dict[str, Any]]:
    """Times at which a player leaves a sustained behind-baseline hold.

    A serve is preceded by the server standing still behind their own baseline.  The end of
    such a hold is the only automatic mark this module has for "a new attempt starts here".
    """
    marks: list[dict[str, Any]] = []
    for side, (times, court, speed) in motion.items():
        outward = -1.0 if side == "near" else 1.0
        held = (
            (outward * (court[:, 1] - BASELINE_COURT_Y[side]) >= SERVE_STANCE_DEPTH_METRES)
            & (speed <= SERVE_STANCE_SPEED)
            & (times >= start_seconds)
            & (times <= end_seconds)
        )
        index = 0
        while index < len(times):
            if not held[index]:
                index += 1
                continue
            stop = index
            while stop + 1 < len(times) and held[stop + 1] and times[stop + 1] - times[stop] < 0.2:
                stop += 1
            if times[stop] - times[index] >= SERVE_STANCE_HOLD_SECONDS:
                marks.append(
                    {
                        "side": side,
                        "seconds": float(times[stop]),
                        "hold_seconds": float(times[stop] - times[index]),
                    }
                )
            index = stop + 1
    return sorted(marks, key=lambda row: row["seconds"])


def second_serve_marks(
    motion: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    peaks: np.ndarray,
    *,
    first_impact_seconds: float,
    start_seconds: float,
    end_seconds: float,
    fps: float,
) -> list[dict[str, Any]]:
    """Serve stances inside one segment that a later racket impact confirms.

    This is the fault-then-second-serve case: one coarse segment holding two attempts.
    """
    low, high = SECOND_SERVE_IMPACT_WINDOW_FRAMES
    confirmed = []
    for mark in serve_stance_marks(motion, start_seconds=start_seconds, end_seconds=end_seconds):
        if mark["seconds"] - first_impact_seconds < SECOND_SERVE_MINIMUM_SEPARATION_SECONDS:
            continue
        if confirmed and mark["seconds"] - confirmed[-1]["seconds"] < (
            SECOND_SERVE_MINIMUM_SEPARATION_SECONDS
        ):
            continue
        window = (
            peaks[
                (peaks[:, 0] >= mark["seconds"] + low / fps)
                & (peaks[:, 0] <= mark["seconds"] + high / fps)
            ]
            if len(peaks)
            else np.empty((0, 2))
        )
        if not len(window):
            continue
        if end_seconds - mark["seconds"] < SECOND_SERVE_MINIMUM_SEPARATION_SECONDS / 2:
            continue
        confirmed.append({**mark, "impact_seconds": float(window[0, 0])})
    return confirmed


# ---------------------------------------------------------------- ledger refinement


def _row_number(row: Mapping[str, Any], key: str) -> float | None:
    return _finite(row.get(key))


def refine_rows(
    rows: Sequence[Mapping[str, Any]],
    motion: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    peaks: np.ndarray,
    *,
    fps: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Refine, split and re-join one match ledger without dropping a coarse segment."""
    expanded: list[dict[str, Any]] = []
    counts = defaultdict(int)
    for index, original in enumerate(rows):
        row = dict(original)
        coarse_start = _row_number(row, "rally_t_start")
        coarse_end = _row_number(row, "rally_t_end")
        if coarse_start is None or coarse_end is None:
            expanded.append({**row, "boundary_source": "coarse_s1_unparsable"})
            counts["unparsable_rows"] += 1
            continue
        contact = _row_number(row, "first_impact_t")
        anchor = contact if contact is not None else coarse_start
        splits = second_serve_marks(
            motion,
            peaks,
            first_impact_seconds=anchor,
            start_seconds=coarse_start,
            end_seconds=coarse_end,
            fps=fps,
        )
        counts["second_serve_splits"] += len(splits)
        attempts = [(coarse_start, contact, None)]
        for mark in splits:
            previous_start, previous_contact, _ = attempts[-1]
            attempts[-1] = (previous_start, previous_contact, mark["seconds"])
            attempts.append((mark["seconds"], mark["impact_seconds"], None))
        for part, (part_start, part_contact, part_ceiling) in enumerate(attempts, start=1):
            ceiling = part_ceiling if part_ceiling is not None else coarse_end
            ending = physical_ending_witness(
                motion,
                peaks,
                first_impact_seconds=part_contact if part_contact is not None else part_start,
                coarse_end_seconds=ceiling,
                fps=fps,
            )
            toss = toss_release_witness(first_impact_seconds=part_contact, fps=fps)
            start = part_start
            if not toss["abstained"]:
                start = min(
                    max(float(toss["seconds"]), part_start - SERVE_MOTION_SECONDS * 2), ceiling
                )
            else:
                counts["toss_abstentions"] += 1
            end = ceiling
            if not ending["abstained"]:
                end = float(ending["seconds"])
            else:
                counts["ending_abstentions"] += 1
            expanded.append(
                {
                    **row,
                    "rally_t_start": round(min(start, end), 3),
                    "rally_t_end": round(max(end, min(start, end) + 1.0 / fps), 3),
                    "first_impact_t": round(part_contact, 3) if part_contact is not None else "",
                    "boundary_source": "|".join(
                        source
                        for source, used in (
                            (f"ending:{ending.get('witness')}", not ending["abstained"]),
                            ("toss:service_motion", not toss["abstained"]),
                            ("coarse_s1", ending["abstained"] and toss["abstained"]),
                        )
                        if used
                    ),
                    "coarse_rally_t_start": round(coarse_start, 3),
                    "coarse_rally_t_end": round(coarse_end, 3),
                    "toss_seconds": round(float(toss["seconds"]), 3)
                    if not toss["abstained"]
                    else "",
                    "toss_abstained": toss["abstained"],
                    "toss_abstention_reason": toss["abstention_reason"] or "",
                    "ending_seconds": round(float(ending["seconds"]), 3)
                    if not ending["abstained"]
                    else "",
                    "ending_confidence": ending["confidence"],
                    "ending_abstained": ending["abstained"],
                    "ending_abstention_reason": ending["abstention_reason"] or "",
                    "ending_snapped_to_impact": ending["snapped_to_impact"],
                    "ending_witness": ending.get("witness") or "",
                    "source_segment_index": index,
                    "source_attempt_part": part,
                    "source_attempt_parts": len(attempts),
                }
            )

    repaired: list[dict[str, Any]] = []
    for row in expanded:
        previous = repaired[-1] if repaired else None
        gap_start = float(previous["rally_t_end"]) - SERVE_MOTION_SECONDS if previous else 0.0
        gap_end = float(row["first_impact_t"] or row["rally_t_start"]) if previous else 0.0
        rejoin = (
            previous is not None
            # Absence of player tracks is not evidence that no serve happened: a rejoin
            # needs the gap to be actually observed.
            and motion_samples(motion, start_seconds=gap_start, end_seconds=gap_end) >= 5
            and int(row.get("source_attempt_part", 1)) == 1
            and int(previous.get("source_attempt_parts", 1))
            == int(previous.get("source_attempt_part", 1))
            and row.get("point_index") == previous.get("point_index")
            and _finite(row["rally_t_start"]) is not None
            and float(row["rally_t_start"]) - float(previous["rally_t_end"])
            < FALSE_SPLIT_MAXIMUM_GAP_SECONDS
            and not serve_stance_marks(motion, start_seconds=gap_start, end_seconds=gap_end)
        )
        if rejoin:
            previous["rally_t_end"] = row["rally_t_end"]
            for key in (
                "ending_seconds",
                "ending_confidence",
                "ending_abstained",
                "ending_abstention_reason",
                "ending_snapped_to_impact",
                "ending_witness",
            ):
                previous[key] = row[key]
            previous["rejoined_source_segments"] = (
                f"{previous.get('rejoined_source_segments') or previous['source_segment_index']}"
                f",{row['source_segment_index']}"
            )
            counts["false_splits_rejoined"] += 1
            continue
        repaired.append(row)
    # Pulling a toss back one service motion can cross the previous attempt's ending.
    # Attempt spans must stay disjoint, so the earlier ending wins.
    for position, row in enumerate(repaired, start=1):
        row["pt"] = position
        if position == 1:
            continue
        previous_end = float(repaired[position - 2]["rally_t_end"])
        if float(row["rally_t_start"]) < previous_end:
            row["rally_t_start"] = round(min(previous_end, float(row["rally_t_end"])), 3)
            row["toss_abstention_reason"] = "clamped_to_previous_attempt_end"
            counts["toss_clamped_to_previous_end"] += 1
    report = {
        "schema": "s1_boundary_refinement_report_v1",
        "coarse_segments": len(rows),
        "refined_segments": len(repaired),
        "counts": dict(counts),
        "configuration": {
            "speed_smoothing_seconds": SPEED_SMOOTHING_SECONDS,
            "rally_speed_metres_per_second": RALLY_SPEED_METRES_PER_SECOND,
            "ending_snap_frames": ENDING_SNAP_FRAMES,
            "short_attempt_window_seconds": SHORT_ATTEMPT_WINDOW_SECONDS,
            "service_motion_seconds": SERVE_MOTION_SECONDS,
            "second_serve_minimum_separation_seconds": SECOND_SERVE_MINIMUM_SEPARATION_SECONDS,
            "false_split_maximum_gap_seconds": FALSE_SPLIT_MAXIMUM_GAP_SECONDS,
        },
        "labels_or_reviewed_inputs": [],
    }
    return repaired, report


def write_ledger(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def refine_match(
    *, s1_dir: Path, evidence_dir: Path, output_dir: Path, fps: float
) -> dict[str, Any]:
    ledger = s1_dir / LEDGER_NAME
    peaks_path = s1_dir / PEAKS_NAME
    tracks_path = evidence_dir / TRACKS_NAME
    with ledger.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    peaks = load_audio_peaks(peaks_path)
    motion = load_player_motion(tracks_path, fps=fps)
    refined, report = refine_rows(rows, motion, peaks, fps=fps)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_ledger(output_dir / LEDGER_NAME, refined)
    report["fps"] = fps
    report["motion_sides"] = sorted(motion)
    report["inputs"] = [
        provenance.file_record(ledger),
        provenance.file_record(peaks_path),
        provenance.file_record(tracks_path),
    ]
    report["provenance"] = provenance.build_provenance(
        root=Path(__file__).resolve().parents[2],
        mode=provenance.AUTOMATIC_MODE,
        configuration=report["configuration"],
        reused_artifacts=report["inputs"],
        fallbacks=[
            {"name": "ending_unavailable", "action": "preserve the coarse S1 rally end"},
            {"name": "toss_unavailable", "action": "preserve the coarse S1 rally start"},
        ],
    )
    report["output"] = provenance.file_record(output_dir / LEDGER_NAME)
    (output_dir / REPORT_NAME).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--s1-root",
        type=Path,
        action="append",
        default=[],
        required=True,
        help="root holding <match>/automatic_point_ledger_v1.csv; repeatable",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        required=True,
        help="post-segmentation evidence root holding <match>/player_tracks_native_v1.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="pipeline manifest naming the matches and their source fps",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    reports = {}
    for match in manifest["matches"]:
        match_id = str(match["id"])
        s1_dir = next(
            (root / match_id for root in args.s1_root if (root / match_id / LEDGER_NAME).is_file()),
            None,
        )
        if s1_dir is None:
            raise FileNotFoundError(f"no automatic S1 ledger for {match_id}")
        reports[match_id] = refine_match(
            s1_dir=s1_dir,
            evidence_dir=args.evidence_root / match_id,
            output_dir=args.output / match_id,
            fps=float(match["source_fps"]),
        )
        print(
            f"[boundaries] {match_id} "
            f"{reports[match_id]['coarse_segments']} -> {reports[match_id]['refined_segments']} "
            f"{reports[match_id]['counts']}",
            flush=True,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / REPORT_NAME).write_text(
        json.dumps(
            {
                "schema": "s1_boundary_refinement_index_v1",
                "matches": {
                    k: {
                        "counts": v["counts"],
                        "refined_segments": v["refined_segments"],
                        "coarse_segments": v["coarse_segments"],
                    }
                    for k, v in reports.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
