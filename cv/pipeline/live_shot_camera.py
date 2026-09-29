"""Scope a clip-level camera hold to the live wide-shot span.

The shot gate holds a whole clip for ``phase:multiple_camera_shots`` or
``no_play_camera_shot`` when a close-up shares the clip with the point. S6's own
per-frame court transport already says which frames have the court on the paint.
This module reads that support and nothing else: no label, no picture, no new
threshold on the registration itself.

A live span is a run of reliable per-frame camera frames, with gaps up to half a
second bridged so a registration flicker is not a close-up. A run shorter than
two seconds is not a wide shot (a one-frame homography is not a view), and the
clip is left exactly as the producer gated it.

Two edits. Both stay off unless the policy names the switch. The production
policy names it:

* release — on a point held only for that shot-composition reason, a physical
  row inside the live span loses the camera hold. A tracking hold on the same
  row stays. Model abstention stays for the operating point to re-read.
* rehold — on any point that has a live span, including one the gate already
  keeps, a physical row outside the span is held. That is the close-up trim.
  The hand-dribble the operating point was promoting at frame 72 sits in the
  close-up before the span, which is why a held contact reached the cascade as
  an accepted emission: ``restate`` clears ``abstain`` on a row that is both
  below the producer threshold and tracking-held, and the inventory reads
  ``abstain`` alone.
"""

from __future__ import annotations

import math

POLICY_KEY = "live_shot_camera_eligibility"
SCHEMA = "live_shot_camera_eligibility_v1"
SHOT_REASONS = frozenset({"no_play_camera_shot", "phase:multiple_camera_shots"})
CAMERA_HOLD = "hard_camera_failure"
OUTSIDE_LIVE_SHOT = "outside_live_wide_shot"
BRIDGE_SECONDS = 0.5
MINIMUM_LIVE_SECONDS = 2.0
PHYSICAL = frozenset({"contact", "bounce", "net_hit"})


def _bridged(spans: list[list[int]] | list[tuple[int, int]], fps: float) -> list[tuple[int, int]]:
    frames: list[int] = []
    for low, high in spans:
        frames.extend(range(int(low), int(high) + 1))
    ordered = sorted(set(frames))
    if not ordered:
        return []
    gap = max(1, int(math.floor(BRIDGE_SECONDS * fps)))
    runs = [[ordered[0], ordered[0]]]
    for frame in ordered[1:]:
        if frame - runs[-1][1] <= gap:
            runs[-1][1] = frame
        else:
            runs.append([frame, frame])
    return [(low, high) for low, high in runs]


def live_spans(supported_spans, fps: float) -> list[tuple[int, int]] | None:
    """The wide-shot runs, or None when the support is not a wide shot.

    None means "do not touch this clip". An empty list is not used: no support
    and a flicker are the same decision, leave the producer gate alone.
    """

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("live-shot camera eligibility needs a positive fps")
    runs = _bridged(list(supported_spans or []), fps)
    covered = sum(high - low + 1 for low, high in runs)
    if covered < MINIMUM_LIVE_SECONDS * fps:
        return None
    return runs


def covers(spans: list[tuple[int, int]] | None, frame: float, fps: float) -> bool:
    """True when ``frame`` is inside a live run or within the continuity bridge of one.

    The homography often locks a few frames after the cut back to the wide shot.
    A contact in that gap is still the wide shot. A close-up, which lasts seconds,
    stays outside.
    """

    if spans is None or not math.isfinite(frame) or not math.isfinite(fps) or fps <= 0:
        return False
    margin = BRIDGE_SECONDS * fps
    return any(low - margin <= frame <= high + margin for low, high in spans)


def scope_event_spans(record: dict, supported_spans) -> list:
    """The frames the event model may see for one active-play record.

    The producer leaves ``event_spans`` empty when shot segmentation finds no
    play camera, and the event model then builds no crops: that is the
    short-circuit in front of S5. A clip refused only for
    ``no_play_camera_shot`` whose per-frame court support is a wide shot is
    scoped to that shot. Anything the producer already scoped, and any other
    refusal, is returned unchanged. Callers pass the producer's spans straight
    through unless the live-shot switch is on.
    """

    existing = [list(span) for span in record.get("event_spans") or []]
    if existing or list(record.get("reasons") or []) != ["no_play_camera_shot"]:
        return existing
    live = live_spans(supported_spans, float(record["fps"]))
    if not live:
        return existing
    return [[float(low), float(high)] for low, high in live]


def shot_composition_hold(gate_row: dict | None) -> bool:
    """True when the point gate's only camera failure is the clip-level shot veto."""

    if not isinstance(gate_row, dict) or gate_row.get("decision") == "retain":
        return False
    if "hard_camera_failure" not in (gate_row.get("reasons") or []):
        return False
    if gate_row.get("hard_timing_failure") is True or gate_row.get("court_geometry_valid") is False:
        return False
    active = set(gate_row.get("active_play_reasons") or [])
    return bool(active) and active <= SHOT_REASONS


def _same_clip(row: dict, match_id: str, clip: str) -> bool:
    if row.get("match_id") != match_id:
        return False
    name = str(row.get("clip", ""))
    return name == clip or name.endswith("__" + clip)


def release_camera_holds(
    rows: list[dict],
    *,
    match_id: str,
    clip: str,
    spans: list[tuple[int, int]] | None,
    release: bool,
    fps: float,
) -> tuple[list[dict], dict]:
    """Drop a shot-composition camera hold on rows inside the live span.

    Rows outside the span, rows on a point that is not a shot-composition hold,
    and a clip with no wide shot are copied through unchanged.
    """

    census = {"schema": SCHEMA, "phase": "release", "released": 0, "tracking_kept": 0}
    if not release or spans is None:
        return rows, census
    output = []
    for source in rows:
        if (
            not _same_clip(source, match_id, clip)
            or source.get("event_type") not in PHYSICAL
            or CAMERA_HOLD not in (source.get("point_gate_failure_reasons") or [])
            or not covers(spans, float(source.get("frame", float("nan"))), fps)
        ):
            output.append(source)
            continue
        reasons = [
            reason
            for reason in source.get("point_gate_failure_reasons") or []
            if reason != CAMERA_HOLD
        ]
        tracking = "tracking_arc_abstained" in reasons
        row = {
            **source,
            "point_gate_failure_reasons": reasons,
            "live_shot_camera": {
                "schema": SCHEMA,
                "action": "tracking_hold_kept" if tracking else "camera_hold_released",
                "live_spans": [list(span) for span in spans],
            },
        }
        if tracking:
            census["tracking_kept"] += 1
            # The clip verdict was a hold only because of the shot veto. With
            # that reason gone, a tracking-only row is the retained-point shape
            # the serve release already accepts, and it stays gate-held.
            if reasons == ["tracking_arc_abstained"]:
                row["point_gate_verdict"] = "retain"
        else:
            row["gate_held"] = False
            row["point_gate_verdict"] = "retain"
            if source.get("model_abstain") is False:
                row["abstain"] = False
            census["released"] += 1
        output.append(row)
    return output, census


def rehold_closeups(
    rows: list[dict],
    *,
    match_id: str,
    clip: str,
    spans: list[tuple[int, int]] | None,
    fps: float,
) -> tuple[list[dict], dict]:
    """Hold physical rows outside the live span, including on a clip the gate keeps.

    A clip without a wide shot is unchanged, so a fallback camera that never
    registers a run cannot empty a point the gate retained.
    """

    census = {"schema": SCHEMA, "phase": "rehold", "reheld": 0}
    if spans is None:
        return rows, census
    output = []
    for source in rows:
        frame = source.get("frame")
        if (
            not _same_clip(source, match_id, clip)
            or source.get("event_type") not in PHYSICAL
            or source.get("abstain") is not False
            or covers(spans, float(frame), fps)
        ):
            output.append(source)
            continue
        reasons = list(source.get("point_gate_failure_reasons") or [])
        if OUTSIDE_LIVE_SHOT not in reasons:
            reasons.append(OUTSIDE_LIVE_SHOT)
        output.append(
            {
                **source,
                "abstain": True,
                "gate_held": True,
                "point_gate_verdict": "hold",
                "point_gate_failure_reasons": reasons,
                "live_shot_camera": {
                    "schema": SCHEMA,
                    "action": "closeup_reheld",
                    "live_spans": [list(span) for span in spans],
                },
            }
        )
        census["reheld"] += 1
    return output, census
