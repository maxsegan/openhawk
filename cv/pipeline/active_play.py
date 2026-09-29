#!/usr/bin/env python3
"""Trim a point clip to its active-play span, with an explicit per-point verdict.

Point clips carry three kinds of non-play footage, and each needs a different detector:

  CUTAWAYS    a different camera entirely (a face close-up, the crowd). Detected by
              `shot_segments.py`: frames of the play shot register to one another by
              homography; other shots do not. Validated 255/255 labeled events preserved.
  DEAD TIME   play-camera footage between rallies -- ball girls collecting balls, the server
              settling. Detected by `play_phase.py` from ball behaviour (traversal, net
              crossings, speed). Validated 0.977 rally-event safety on accepted points.
  PADDING     the slack before the serve and after the terminal event inside the rally span
              itself. Only trimmable once event anchors exist, so it is an OPTIONAL input
              here rather than a second inference.

This module composes those into one decision per clip: the active span(s), a trim window,
and a verdict. Under the owner's operating point (2026-07-26) up to ~20% of points may be
invalidated outright provided the defect is DETECTED, so ambiguity is declared, never
guessed.

TRIM SEMANTICS ARE CONSERVATIVE BY DESIGN. The trim window is the bounding interval of all
accepted active spans plus a margin; interior dead time is annotated but NOT cut, because
dead-ball rebounds are owner-adjudicated truth and phase is a label, never a filter
(AGENT_DIRECTIVE.md). Downstream consumers get the annotation and decide for themselves.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from cv.pipeline import resolution as res
from cv.pipeline.play_phase import detect_phase
from cv.pipeline.shot_segments import ShotSegmentation, segment_shots

__all__ = ["ActivePlayResult", "apply_leakage_trims", "resolve_active_play"]

# Generous by design: the phase span's hysteresis lags the final event slightly (measured
# one event 3 frames outside a 0.6 s margin), and trimming exists to cut long dead heads and
# tails, not to shave fractions of a second.
TRIM_MARGIN_SECONDS = 1.2
# Terminal events (the point-ending bounce) often occur after rally activity decays
# below the phase hysteresis, so the raw final span can end BEFORE the last live event
# (measured: 14/87 truthed clips had a negative trailing buffer, median 6 frames).
# Extend the final span toward trailing ball-track evidence by at most this much.
TRAILING_TRACK_EXTENSION_SECONDS = 1.0
GAP_BRIDGE_MINIMUM_TRACK_COVERAGE = 0.8
GAP_BRIDGE_MICRO_GAP_SECONDS = 0.25
# Phase hysteresis can end one or two native frames before a terminal collision.
# Protect that event without re-admitting the longer post-point tail.
# 0.12 left terminal bounces 0.2 s past the edge outside event scope (2/822 truth
# events on the frozen corpus); 0.25 covers them at +6 frames of decoder scope.
EVENT_SCOPE_MARGIN_SECONDS = 0.25
NET_GUARDED_PHASE_PADDING_SECONDS = 0.16
# The shot gate's safety was validated at this stride on the audit cohort.
SHOT_STRIDE = 4
MINIMUM_LEAKAGE_TRIM_SECONDS = 2.0


@dataclass
class ActivePlayResult:
    clip: str
    fps: float
    n_frames: int
    active_spans: list[tuple[float, float]] = field(default_factory=list)
    event_spans: list[tuple[float, float]] = field(default_factory=list)
    trim: tuple[float, float] | None = None
    trimmed_fraction: float = 0.0
    point_valid: bool = True
    reasons: list[str] = field(default_factory=list)
    shot_summary: list[dict] = field(default_factory=list)
    phase_reason: str = ""
    phase_profile: str = "baseline"
    native_cut_continuity: bool = False
    boundary_continuity: list[dict] = field(default_factory=list)
    court_anchor_camera: dict | None = None

    def contains(self, frame: float) -> bool:
        return self.trim is not None and self.trim[0] <= frame <= self.trim[1]

    def contains_event(self, frame: float) -> bool:
        return any(start <= frame <= end for start, end in self.event_spans)

    def as_dict(self) -> dict:
        entry = {
            "clip": self.clip,
            "fps": self.fps,
            "n_frames": self.n_frames,
            "active_spans": [[float(a), float(b)] for a, b in self.active_spans],
            "event_spans": [[float(a), float(b)] for a, b in self.event_spans],
            "trim": [float(self.trim[0]), float(self.trim[1])] if self.trim else None,
            "trimmed_fraction": round(self.trimmed_fraction, 4),
            "point_valid": self.point_valid,
            "reasons": self.reasons,
            "phase_reason": self.phase_reason,
            "phase_profile": self.phase_profile,
            "shots": self.shot_summary,
        }
        # Declared only when selected, so a default entry keeps its current keys and values.
        if self.native_cut_continuity:
            entry["native_cut_continuity"] = True
            entry["boundary_continuity"] = self.boundary_continuity
        if self.court_anchor_camera is not None:
            entry["court_anchor_camera"] = self.court_anchor_camera
        return entry


def _subtract_trim_windows(
    spans: list[list[float]], trim_windows: list[list[float]]
) -> list[list[float]]:
    """Subtract inclusive leakage windows from disjoint closed-frame spans."""
    remaining = [[float(start), float(end)] for start, end in spans]
    for trim_start, trim_end in sorted((float(start), float(end)) for start, end in trim_windows):
        updated = []
        for span_start, span_end in remaining:
            if trim_end < span_start or trim_start > span_end:
                updated.append([span_start, span_end])
                continue
            if span_start < trim_start:
                updated.append([span_start, min(span_end, trim_start - 1.0)])
            if span_end > trim_end:
                updated.append([max(span_start, trim_end + 1.0), span_end])
        remaining = updated
    return remaining


def _court_only_visual_run(entry: dict, run: dict) -> bool:
    """A supported interior picture run need not inherit a calibration veto."""
    if run.get("reasons") != ["court_support_lost"] or entry.get("point_valid") is not True:
        return False
    start, end = float(run["start_frame"]), float(run["end_frame"])
    support = run.get("visual_play_support", {})
    if (
        support.get("schema") != "native_two_player_visual_support_v1"
        or support.get("supported") is not True
        or support.get("start_frame") != start
        or support.get("end_frame") != end
    ):
        return False
    # Strict containment preserves shot boundaries, including uncertain cut gaps.
    shots = [
        shot
        for shot in entry.get("shots", [])
        if shot.get("is_play_camera") is True
        and float(shot["start_frame"]) < start <= end < float(shot["end_frame"])
    ]
    return len(shots) == 1


def apply_leakage_trims(entry: dict, proposal: dict, *, court_loss_policy: str = "strict") -> dict:
    """Apply automatic camera-leakage proposals to active and event scope.

    This is the production adoption of ``cv/experiments/leakage_repair/measure.py``.
    The experiment's closed-frame subtraction and two-second minimum are preserved;
    a proposal that would leave less evidence is held without materializing the trim.
    """
    if court_loss_policy not in ("strict", "preserve_supported_play"):
        raise ValueError("unsupported court loss policy")
    output = dict(entry)
    retained = [
        row
        for row in proposal.get("proposed_trims", [])
        if court_loss_policy == "preserve_supported_play" and _court_only_visual_run(entry, row)
    ]
    trims = [
        [float(row["start_frame"]), float(row["end_frame"])]
        for row in proposal.get("proposed_trims", [])
        if row not in retained
    ]
    active_before = [list(map(float, span)) for span in entry.get("active_spans", [])]
    event_before = [
        list(map(float, span)) for span in entry.get("event_spans", entry.get("active_spans", []))
    ]
    active_after = _subtract_trim_windows(active_before, trims)
    fps = float(entry.get("fps") or proposal.get("fps") or 0.0)
    remaining_frames = sum(end - start + 1.0 for start, end in active_after)
    remaining_seconds = remaining_frames / fps if fps > 0.0 else 0.0
    applied = bool(trims and active_after != active_before)
    hold = applied and remaining_seconds < MINIMUM_LEAKAGE_TRIM_SECONDS

    if applied and not hold:
        output["active_spans"] = active_after
        output["event_spans"] = _subtract_trim_windows(event_before, trims)
        if active_after:
            output["trim"] = [active_after[0][0], active_after[-1][1]]
            n_frames = max(int(output.get("n_frames") or 0), 1)
            kept_frames = sum(end - start + 1.0 for start, end in active_after)
            output["trimmed_fraction"] = round(max(0.0, 1.0 - kept_frames / n_frames), 4)
    elif hold:
        output["point_valid"] = False
        reasons = list(output.get("reasons", []))
        if "leakage_trim_below_2s" not in reasons:
            reasons.append("leakage_trim_below_2s")
        output["reasons"] = reasons

    output["leakage_trim"] = {
        "source_schema": proposal.get("schema", "play_camera_leakage_v1"),
        "decision": "hold" if hold else ("applied" if applied else "no_change"),
        "minimum_remaining_seconds": MINIMUM_LEAKAGE_TRIM_SECONDS,
        "remaining_seconds_if_applied": round(remaining_seconds, 6),
        "proposed_trims": proposal.get("proposed_trims", []),
        "proposal_active_spans": proposal.get("active_spans", active_before),
        "proposal_spans_after_trims": proposal.get("proposed_spans", active_after),
        "active_spans_before": active_before,
        "active_spans_after": active_after,
    }
    if court_loss_policy != "strict":
        output["leakage_trim"]["visual_scope_policy"] = {
            "policy": court_loss_policy,
            "retained_court_only_runs": retained,
            "camera_support": "unchanged; court-line loss remains unresolved",
            "competitive_phase_inferred": False,
            "scope": "preserve original visual observations within existing phase and shot scope",
        }
    return output


def _intersect_spans_with_play_shots(
    spans: list[tuple[float, float]], segmentation: ShotSegmentation
) -> list[tuple[float, float]]:
    """Clip rally spans to play-camera shots so a cutaway can never sit inside a span."""
    play_shots: list[tuple[float, float]] = []
    for shot in segmentation.shots:
        if shot.get("is_play_camera"):
            play_shots.append((float(shot["start_frame"]), float(shot["end_frame"])))
    out: list[tuple[float, float]] = []
    for start, end in spans:
        for shot_start, shot_end in play_shots:
            lo, hi = max(start, shot_start), min(end, shot_end)
            if hi > lo:
                out.append((lo, hi))
    return sorted(out)


def _event_spans(
    spans: list[tuple[float, float]],
    segmentation: ShotSegmentation,
    *,
    fps: float,
    margin_seconds: float = EVENT_SCOPE_MARGIN_SECONDS,
) -> list[tuple[float, float]]:
    """Expand rally evidence slightly without crossing a play-camera boundary."""
    margin = margin_seconds * fps
    expanded = [(max(0.0, start - margin), end + margin) for start, end in spans]
    clipped = _intersect_spans_with_play_shots(expanded, segmentation)
    merged: list[tuple[float, float]] = []
    # A hole of a few frames between clipped sub-spans is shot-segmentation stutter
    # (a real cutaway lasts hundreds of frames); truth contacts land inside such
    # holes, so merge across anything up to the micro-gap width.
    merge_gap = max(1.0, GAP_BRIDGE_MICRO_GAP_SECONDS * fps)
    for start, end in clipped:
        if merged and start <= merged[-1][1] + merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _bridge_track_continuous_gaps(
    spans: list[tuple[float, float]],
    track: np.ndarray,
    fps: float,
) -> list[tuple[float, float]]:
    """Merge adjacent spans whose gap the ball track covers almost continuously.

    A rally does not pause: a span split whose gap still shows the tracked ball on
    nearly every frame is segmentation stutter, and truth events land inside such
    gaps (measured 2/9 remaining outside events on the frozen corpus). Gaps without
    track continuity (real between-attempt lulls) are preserved.
    """
    if len(spans) < 2 or track is None or len(track) == 0:
        return spans
    frames = set(int(f) for f in track[:, 0])
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        gap = range(int(merged[-1][1]) + 1, int(start))
        coverage = sum(1 for f in gap if f in frames) / len(gap) if len(gap) else 1.0
        micro_gap = len(gap) <= GAP_BRIDGE_MICRO_GAP_SECONDS * fps
        if micro_gap or coverage >= GAP_BRIDGE_MINIMUM_TRACK_COVERAGE:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _extend_final_span_to_track(
    spans: list[tuple[float, float]],
    track: np.ndarray,
    fps: float,
    segmentation,
) -> list[tuple[float, float]]:
    """Extend the final span's trailing edge to cover trailing ball-track evidence.

    The terminal bounce frequently lands just after rally activity drops below the
    phase hysteresis. Extension is evidence-bound (last observed track frame within
    the window) and never crosses out of the play shot containing the span end.
    """
    if not spans or track is None or len(track) == 0:
        return spans
    start, end = spans[-1]
    limit = end + TRAILING_TRACK_EXTENSION_SECONDS * fps
    frames = track[:, 0]
    trailing = frames[(frames > end) & (frames <= limit)]
    if len(trailing) == 0:
        return spans
    new_end = float(trailing.max())
    for shot in segmentation.shots:
        if shot.get("is_play_camera") and shot["start_frame"] <= end <= shot["end_frame"]:
            new_end = min(new_end, float(shot["end_frame"]))
            break
    if new_end > end:
        spans = spans[:-1] + [(start, new_end)]
    return spans


def resolve_active_play(
    frame_paths: list[Path],
    track: np.ndarray,
    fps: float,
    net_row: np.ndarray | None = None,
    event_frames: list[float] | None = None,
    trim_margin_seconds: float = TRIM_MARGIN_SECONDS,
    shot_stride: int = SHOT_STRIDE,
    phase_profile: Literal["baseline", "net_guarded_v1"] = "baseline",
    native_cut_continuity: bool = False,
    court_anchor: Path | None | Literal[False] = False,
) -> ActivePlayResult:
    """Compose shot gating and phase detection into one trim decision.

    `track` is (n, >=3) frame/x/y in artifact space. `event_frames`, when provided (labels
    in validation, proposals in production), tighten the trim to serve-through-terminal;
    they are never required.

    `native_cut_continuity` is an explicit segmentation option: sampled cut proposals that
    native frames show to be continuous motion are withdrawn before this module reads the
    shots. Nothing here changes its own thresholds or verdict logic; the revised
    segmentation simply feeds the existing play-shot count and span intersection.

    `court_anchor` selects `court_anchor_camera.reanchor`: the court fit's anchor
    picture (None when the fit abstained) decides the play camera where the shot gate's
    longest-shot reference disagrees with it. False, the default, leaves it off.
    """
    clip = frame_paths[0].parent.name if frame_paths else "unknown"
    result = ActivePlayResult(
        clip=clip,
        fps=fps,
        n_frames=len(frame_paths),
        phase_profile=phase_profile,
        native_cut_continuity=native_cut_continuity,
    )
    if len(frame_paths) < 10 or track is None or len(track) < 10:
        result.point_valid = False
        result.reasons.append("insufficient_input")
        return result

    # Forwarded only when selected, so an unchanged default call reaches segment_shots
    # with its existing arguments.
    segmentation = (
        segment_shots(frame_paths, fps=fps, stride=shot_stride, native_cut_continuity=True)
        if native_cut_continuity
        else segment_shots(frame_paths, fps=fps, stride=shot_stride)
    )
    if court_anchor is not False:
        from cv.pipeline.court_anchor_camera import reanchor
        from cv.pipeline.shot_segments import DEFAULT_MINIMUM_SHOT_SECONDS

        segmentation, result.court_anchor_camera = reanchor(
            segmentation,
            frame_paths,
            court_anchor,
            fps=fps,
            stride=shot_stride,
            minimum_shot_seconds=DEFAULT_MINIMUM_SHOT_SECONDS,
        )
    result.shot_summary = segmentation.shots
    if native_cut_continuity:
        result.boundary_continuity = list(getattr(segmentation, "boundary_continuity", []))
    play_shot_count = sum(1 for shot in segmentation.shots if shot.get("is_play_camera"))
    if play_shot_count == 0:
        result.point_valid = False
        result.reasons.append("no_play_camera_shot")
        return result

    # Pass the PLAY-shot count, not the raw shot count. Cutaway shots are identified and
    # excluded by the registration gate, so their existence is handled risk, not ambiguity.
    # What remains genuinely hard (the owner's stated case) is the play camera itself
    # changing mid-point -- i.e. more than two play shots fragmenting the timeline.
    if phase_profile == "net_guarded_v1" and net_row is not None:
        net_phase = detect_phase(
            track[:, 0],
            track[:, 1:3],
            net_row,
            fps=fps,
            shot_count=play_shot_count,
        )
        if net_phase.point_valid and net_phase.spans:
            padding = NET_GUARDED_PHASE_PADDING_SECONDS * fps
            net_phase.spans = [
                (
                    max(float(track[0, 0]), start - padding),
                    min(float(track[-1, 0]), end + padding),
                )
                for start, end in net_phase.spans
            ]
            phase = net_phase
        else:
            phase = detect_phase(
                track[:, 0],
                track[:, 1:3],
                None,
                fps=fps,
                shot_count=play_shot_count,
            )
            result.phase_profile = "net_guarded_v1:baseline_fallback"
    else:
        phase = detect_phase(
            track[:, 0],
            track[:, 1:3],
            None,
            fps=fps,
            shot_count=play_shot_count,
        )
    result.phase_reason = phase.reason
    if not phase.point_valid:
        result.point_valid = False
        result.reasons.append(f"phase:{phase.reason}")
        # Still emit the spans we saw: an invalidated point goes to review WITH evidence.

    spans = _intersect_spans_with_play_shots(phase.spans, segmentation)
    spans = _extend_final_span_to_track(spans, track, fps, segmentation)
    spans = _bridge_track_continuous_gaps(spans, track, fps)
    result.active_spans = spans
    if not spans:
        result.point_valid = False
        if "phase:no_rally_span_found" not in result.reasons:
            result.reasons.append("no_active_span")
        return result
    result.event_spans = _event_spans(
        spans,
        segmentation,
        fps=fps,
    )

    lo = min(start for start, _ in spans)
    hi = max(end for _, end in spans)
    if event_frames:
        # Anchor-informed tightening: never *extend* beyond the phase evidence, and always
        # keep every known event inside the window.
        lo = min(lo, min(event_frames))
        hi = max(hi, max(event_frames))
    margin = trim_margin_seconds * fps
    first_frame = float(track[0, 0])
    last_frame = float(track[-1, 0])
    trim_lo = max(first_frame, lo - margin)
    trim_hi = min(last_frame, hi + margin)
    # The margin must not back the window out of the play camera into an adjacent cutaway:
    # a trim that OPENS on a face close-up is wrong even if downstream would ignore it.
    # Clamp each end to the play shot containing (or nearest to) the span boundary.
    for shot in segmentation.shots:
        if not shot.get("is_play_camera"):
            continue
        start, end = float(shot["start_frame"]), float(shot["end_frame"])
        if start <= lo <= end:
            trim_lo = max(trim_lo, start)
        if start <= hi <= end:
            trim_hi = min(trim_hi, end)
    trim = (trim_lo, trim_hi)
    result.trim = trim
    total = max(last_frame - first_frame, 1.0)
    result.trimmed_fraction = float(max(0.0, total - (trim[1] - trim[0])) / total)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames-dir", type=Path, required=True, help="directory of pt*/f_*.jpg clips"
    )
    parser.add_argument(
        "--track-csv",
        type=Path,
        required=True,
        help="track CSV: clip,frame,x,y[,...] in artifact space",
    )
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import csv as csv_module
    from collections import defaultdict

    rows: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    with args.track_csv.open() as handle:
        reader = csv_module.DictReader(handle)
        read_point, _, _ = res.native_point_reader(args.track_csv, reader.fieldnames or ())
        for row in reader:
            digits = "".join(ch for ch in row.get("frame", "") if ch.isdigit())
            if digits:
                x, y = read_point(row)
                rows[row["clip"]].append((float(int(digits)), x, y))

    output = {}
    for clip_dir in sorted(args.frames_dir.glob("pt*")):
        paths = sorted(clip_dir.glob("f_*.jpg"))
        track = np.array(sorted(rows.get(clip_dir.name, [])), dtype=float)
        decision = resolve_active_play(paths, track, fps=args.fps)
        output[clip_dir.name] = decision.as_dict()
        span_note = (
            f"trim [{decision.trim[0]:.0f}, {decision.trim[1]:.0f}]" if decision.trim else "no trim"
        )
        print(
            f"{clip_dir.name}: valid={decision.point_valid} {span_note} "
            f"cut={decision.trimmed_fraction:.0%} {';'.join(decision.reasons)}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
