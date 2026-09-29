"""Optional post-S5 service/reset proposals with immutable parent observation ownership.

Reuses existing stance and track-continuity primitives. No model calls, inferred
physical impacts, or fitted states. A proposal is not a certified point ending.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np

from cv.pipeline import segment_boundaries
from cv.pipeline.bounce_detect import split_runs

SCHEMA = "automatic_service_attempt_scope_v1"


def plan(
    *,
    emissions: list[dict],
    observations: list[dict],
    players: Path,
    cameras: dict,
    source_start: float,
    fps: float,
    native_window: tuple[int, int],
    clip: str | None = None,
) -> dict:
    """Produce deterministic child scopes; never silently drop accepted events."""
    from cv.experiments.connected_shooting.auto_packet import (
        automatic_event_epoch,
        automatic_event_interval,
    )

    physical = []
    for index, row in enumerate(emissions):
        if row.get("abstain", False) or row["event_type"] not in {"contact", "bounce", "net_hit"}:
            continue
        epoch = automatic_event_epoch(row)
        interval, _ = automatic_event_interval(row, epoch)
        physical.append(dict(index=index, epoch=epoch, interval=interval, kind=row["event_type"]))
    physical.sort(key=lambda row: (row["epoch"], row["kind"]))
    contacts = [r for r in physical if r["kind"] == "contact"]
    original = dict(
        schema=SCHEMA,
        mode="on",
        native_window=list(native_window),
        source_start_seconds=source_start,
        native_fps=fps,
        original_emissions=deepcopy(emissions),
        proposals=[],
        rejected=[],
        children=[],
        physical_ending_inferred=False,
        runtime_model_calls=0,
    )
    if not contacts:
        return original
    clock_options = {}
    if clip is not None:
        clock = {int(row["frame"]): float(row["native_pts_seconds"]) for row in observations}
        if len(clock) != len(observations):
            raise ValueError("unique source-bound native frame times required")
        # Existing stance predicates use this declared constant native clock.
        # Refuse inconsistent cadence rather than inventing source epochs.
        if any(abs(t - (source_start + (f - 1) / fps)) > 0.000501 for f, t in clock.items()):
            raise ValueError("native PTS differs from declared source window clock")
        clock_options = dict(clip=clip, native_pts_by_frame=clock)
        original["player_motion_clock"] = dict(
            source="verified native image source PTS",
            clip=clip,
            native_frame_count=len(clock),
            missing_t="join by exact clip and native frame",
            existing_t="preserved after millisecond-precision consistency check",
            source_player_rows_modified=False,
        )
    motion = segment_boundaries.load_player_motion(players, fps=fps, **clock_options)
    supported = {int(r["frame"]) for r in cameras["cameras"] if r.get("supported") is True}
    scoped_motion = {}
    for side, (times, xy, speeds) in motion.items():
        frames = np.rint((times - source_start) * fps + 1).astype(int)
        mask = np.asarray([int(f) in supported for f in frames])
        if mask.sum() >= 5:
            scoped_motion[side] = (times[mask], xy[mask], speeds[mask])
    motion = scoped_motion
    original["reliable_camera_frames"] = len(supported)

    def time(frame):
        return source_start + (frame - 1) / fps

    marks = segment_boundaries.second_serve_marks(
        motion,
        np.asarray([[time(c["epoch"]), 1.0] for c in contacts]),
        first_impact_seconds=time(contacts[0]["epoch"]),
        start_seconds=time(native_window[0]),
        end_seconds=time(native_window[1]),
        fps=fps,
    )
    visible = sorted(
        [r for r in observations if r["status"] == "visible"], key=lambda r: r["frame"]
    )
    runs = []
    if visible:
        frames = np.asarray([r["frame"] for r in visible])
        xs = np.asarray([r["x1080"] for r in visible], float)
        ys = np.asarray([r["y1080"] for r in visible], float)
        # Existing split_runs uses55px in the legacy960-wide domain. These
        # source rows are explicitly native1920-wide; scale the same threshold.
        runs = [
            [int(frames[g[0]]), int(frames[g[-1]])]
            for g in split_runs(frames, xs, ys, jump_px=110.0)
        ]
    original["continuity_components"] = runs
    boundaries = []
    for mark in marks:
        current = next(c for c in contacts if time(c["epoch"]) == mark["impact_seconds"])
        previous = [r for r in physical if r["epoch"] < current["epoch"]]
        prior = previous[-1] if previous else None
        gap = (current["interval"][0] - prior["interval"][1]) / fps if prior else None
        left = next((r for r in runs if r[0] <= prior["epoch"] <= r[1]), None) if prior else None
        right = next((r for r in runs if r[0] <= current["epoch"] <= r[1]), None)
        reason = None
        if prior is None or prior["kind"] not in {"net_hit", "bounce"}:
            reason = "previous_live_contact_or_no_terminal_interaction"
        elif gap < segment_boundaries.SECOND_SERVE_MINIMUM_SEPARATION_SECONDS:
            reason = "short_or_uncertain_event_free_gap"
        elif left is None or right is None or left == right:
            reason = "no_supported_identity_break_during_reset"
        elif left[1] <= prior["interval"][1] or right[0] > current["interval"][0]:
            reason = "component_does_not_cover_original_event_interval_and_tail"
        receipt = dict(
            mark=mark,
            contact=deepcopy(current),
            prior=deepcopy(prior),
            minimum_event_gap_seconds=gap,
            previous_component=left,
            next_component=right,
            physical_ending_certified=False,
        )
        if reason:
            original["rejected"].append(dict(receipt, reason=reason))
        else:
            original["proposals"].append(receipt)
            boundaries.append((current, left, right))
    if not boundaries:
        return original
    cuts = [r[0]["epoch"] for r in boundaries]
    for child_index in range(len(boundaries) + 1):
        low = -math.inf if child_index == 0 else cuts[child_index - 1]
        high = math.inf if child_index == len(boundaries) else cuts[child_index]
        assigned = [r for r in physical if low <= r["epoch"] < high]
        start = native_window[0] if child_index == 0 else boundaries[child_index - 1][2][0]
        end = native_window[1] if child_index == len(boundaries) else boundaries[child_index][1][1]
        hold = None
        if not assigned or assigned[0]["kind"] != "contact":
            hold = "child_has_no_originating_contact"
        elif any(not start <= r["interval"][0] <= r["interval"][1] <= end for r in assigned):
            hold = "child_scope_would_drop_original_physical_event_interval"
        # Explicit membership uses original emitted epochs, independently of
        # the model horizon. Refused rows remain in the full parent sidecar.
        indices = [i for i, r in enumerate(emissions) if low <= automatic_event_epoch(r) < high]
        child = dict(
            child_index=child_index + 1,
            original_event_indices=[r["index"] for r in assigned],
            original_emission_indices=indices,
            modeled_native_window=[start, end],
            video_context_native_window=list(native_window),
            hold_reason=hold,
            aftermath=dict(
                kind="parent_visible_scope"
                if child_index == len(boundaries)
                else "identity_censored",
                horizon_frame=end,
                physical_ending_certified=False,
                old_ball_fully_visible_until_horizon_certified=False,
                reason="original_parent_context"
                if child_index == len(boundaries)
                else "original_continuity_break_during_supported_reset",
            ),
            context_is_not_trajectory=True,
        )
        original["children"].append(child)
    flattened = [i for child in original["children"] for i in child["original_event_indices"]]
    if sorted(flattened) != sorted(r["index"] for r in physical):
        raise ValueError("service refinement lost or duplicated original physical events")
    return original
