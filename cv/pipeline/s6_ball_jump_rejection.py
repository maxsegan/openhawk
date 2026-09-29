"""Label-free rejection of short wrong-object islands in an automatic ball track.

Question: can automatic ball rows that sit on another object for a few frames (a shoe, a
line judge) be recognised from the track alone, without labels, detector retraining or the
fitter?  Default off: `apply` is a no-op unless the shared policy key
`automatic_ball_jump_rejection` is declared "on" (absent and "off" are identical).

A passive ball's image position is continuous.  A racket contact or a bounce changes its
velocity once, and the new speed then persists.  A *teleport* is a single inter-row step that
is large in absolute terms and several times faster than the steps on both sides of it: the
ball would have to accelerate and immediately decelerate within two exposures.  An *island*
is a short run of rows that starts (or, read backwards, ends) at a teleport, where the rows
either side of it already agree with each other: the straight join across the island has the
same image velocity as the undisturbed rows before it and after it, and every island row
stands well off that join.  Only such islands are rejected.  Anything less certain is kept:
a run whose far side lies beyond a long detector gap or does not continue the near side, a
run at a racket contact (the velocities either side differ), or a jump with no return.  The
track alone cannot say which side is the ball there.

Usage:
    flags = island_flags([(frame, x, y), ...], fps)      # -> {frame: reason}
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


FIELD = "automatic_ball_jump_rejection"
DEFAULT = "off"
#: "gap_steps" is the on rule plus the gap-step extension (teleports across short detector
#: gaps); it is a separate declared mode so the measured "on" behaviour never moves.
GAP_STEPS = "gap_steps"
GAP_STEP_SECONDS = 0.25
MODES = (DEFAULT, "on", GAP_STEPS)
REASON = "teleport_entered_island"
RECEIPT_SCHEMA = "automatic_ball_jump_rejection_v1"


def validate_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError("automatic_ball_jump_rejection must be 'off', 'on' or 'gap_steps'")
    return mode


@dataclass(frozen=True)
class JumpPolicy:
    """Thresholds in 1080p pixels and seconds; frame counts are derived from the cadence."""

    #: a teleport moves at least this far in one inter-row step
    minimum_jump_px: float = 40.0
    #: and is at least this many times faster than the faster neighbouring step
    speed_ratio: float = 3.0
    #: neighbouring steps slower than this are treated as this (detector jitter floor)
    neighbour_speed_floor_px: float = 4.0
    #: rows more than this far apart in time are not compared (a detector gap, not a step)
    maximum_step_seconds: float = 0.10
    #: an island never spans more than this
    maximum_island_seconds: float = 0.25
    #: and the straight join across it (island plus any detector gap) never more than this
    maximum_join_seconds: float = 0.5
    #: joining the rows outside an island must be at most this fraction of the teleport
    bridge_fraction: float = 0.5
    #: the join's velocity must match the undisturbed velocity on both sides to within
    #: this fraction of the faster one, plus ``velocity_floor_px`` per frame
    velocity_match_fraction: float = 0.35
    velocity_floor_px: float = 6.0
    #: undisturbed rows read on each side (at least ``minimum_side_steps`` steps)
    side_rows: int = 5
    minimum_side_steps: int = 2
    #: every island row stands at least this far from the straight join
    minimum_departure_px: float = 20.0
    #: EXTENSION, off at zero (the measured default).  Above zero, a step across a detector
    #: gap of up to this many seconds may also be a teleport, judged on its implied speed, so
    #: an island that the tracker entered or left through a short gap is a candidate too.  The
    #: acceptance test is unchanged: both undisturbed sides must agree with the join.
    gap_step_seconds: float = 0.0
    #: the join limit that applies instead of ``maximum_join_seconds`` when the extension is on
    gap_join_seconds: float = 0.75

    def validate(self) -> None:
        values = (
            self.minimum_jump_px,
            self.speed_ratio,
            self.neighbour_speed_floor_px,
            self.maximum_step_seconds,
            self.maximum_island_seconds,
            self.maximum_join_seconds,
            self.bridge_fraction,
            self.velocity_match_fraction,
            self.velocity_floor_px,
            self.side_rows,
            self.minimum_side_steps,
            self.minimum_departure_px,
            self.gap_join_seconds,
        )
        if not all(math.isfinite(v) and v > 0 for v in values) or self.bridge_fraction >= 1:
            raise ValueError("finite positive jump-rejection thresholds required")
        if not math.isfinite(self.gap_step_seconds) or self.gap_step_seconds < 0:
            raise ValueError("gap step extension is off at zero or a positive duration")


def _steps(rows: Sequence[tuple[int, float, float]], max_gap: int) -> list[dict | None]:
    """Step k joins rows k-1 and k; None where the rows are too far apart to compare."""
    steps: list[dict | None] = [None]
    for (f0, x0, y0), (f1, x1, y1) in zip(rows[:-1], rows[1:]):
        gap = f1 - f0
        if gap < 1:
            raise ValueError("ordered unique native frames required")
        distance = math.hypot(x1 - x0, y1 - y0)
        steps.append(None if gap > max_gap else dict(distance=distance, speed=distance / gap))
    return steps


def teleports(
    rows: Sequence[tuple[int, float, float]], fps: float, policy: JumpPolicy = JumpPolicy()
) -> list[int]:
    """Indices k such that the step from row k-1 to row k is a teleport."""
    policy.validate()
    max_gap = max(1, round(policy.maximum_step_seconds * fps))
    steps = _steps(rows, max_gap)
    wide = _steps(rows, max(max_gap, round(policy.gap_step_seconds * fps)))
    found = []
    for k in range(1, len(rows)):
        step = wide[k]
        if step is None or step["distance"] < policy.minimum_jump_px:
            continue
        neighbours = [
            steps[j]["speed"]
            for j in (k - 1, k + 1)
            if 0 < j < len(rows) and steps[j] is not None and j != k
        ]
        # Both sides must be measurable: next to a detector gap the track alone cannot tell
        # a teleport from a ball that simply reappeared somewhere else.
        if len(neighbours) != 2:
            continue
        reference = max(policy.neighbour_speed_floor_px, *neighbours)
        if step["speed"] >= policy.speed_ratio * reference:
            found.append(k)
    return found


def _side_velocity(rows, steps, jumps: set[int], start: int, direction: int, policy: JumpPolicy):
    """Mean image velocity over undisturbed rows walking away from ``start``; None if too few."""
    index, used = start, 0
    while used < policy.side_rows - 1:
        step = index + 1 if direction > 0 else index
        if not 0 < step < len(rows) or steps[step] is None or step in jumps:
            break
        index += direction
        used += 1
    if used < policy.minimum_side_steps:
        return None
    first, last = (rows[start], rows[index]) if direction > 0 else (rows[index], rows[start])
    frames = last[0] - first[0]
    return ((last[1] - first[1]) / frames, (last[2] - first[2]) / frames)


def _forward_islands(rows, fps: float, policy: JumpPolicy) -> set[int]:
    """Islands entered by a teleport; the far side is where both sides agree with the join."""
    max_gap = max(1, round(policy.maximum_step_seconds * fps))
    steps = _steps(rows, max_gap)
    wide = _steps(rows, max(max_gap, round(policy.gap_step_seconds * fps)))
    join_limit = (
        policy.gap_join_seconds if policy.gap_step_seconds > 0 else policy.maximum_join_seconds
    ) * fps
    span = policy.maximum_island_seconds * fps
    found = teleports(rows, fps, policy)
    jumps = set(found)
    flagged: set[int] = set()
    skip_until = -1
    for a in found:
        if a <= skip_until:
            continue
        before = rows[a - 1]
        incoming = _side_velocity(rows, steps, jumps, a - 1, -1, policy)
        if incoming is None:
            continue
        for b in range(a + 1, len(rows)):
            after = rows[b]
            if rows[b - 1][0] - rows[a][0] + 1 > span:
                break
            gap = after[0] - before[0]
            if gap > join_limit:
                break
            join = ((after[1] - before[1]) / gap, (after[2] - before[2]) / gap)
            if math.hypot(*join) > policy.bridge_fraction * wide[a]["speed"]:
                continue
            outgoing = _side_velocity(rows, steps, jumps, b, 1, policy)
            if outgoing is None:
                continue
            if any(
                math.hypot(join[0] - side[0], join[1] - side[1])
                > policy.velocity_floor_px
                + policy.velocity_match_fraction * max(math.hypot(*join), math.hypot(*side))
                for side in (incoming, outgoing)
            ):
                continue
            island = rows[a:b]
            if all(
                math.hypot(
                    x - (before[1] + (f - before[0]) * join[0]),
                    y - (before[2] + (f - before[0]) * join[1]),
                )
                >= policy.minimum_departure_px
                for f, x, y in island
            ):
                flagged.update(f for f, _, _ in island)
                skip_until = b
            break
    return flagged


def island_flags(
    rows: Sequence[tuple[int, float, float]], fps: float, policy: JumpPolicy = JumpPolicy()
) -> dict[int, str]:
    """Frames of rows inside a teleport-entered island whose removal restores continuity."""
    rows = sorted((int(f), float(x), float(y)) for f, x, y in rows)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("positive native cadence required")
    policy.validate()
    forward = _forward_islands(rows, fps, policy)
    mirrored = sorted((-f, x, y) for f, x, y in rows)
    backward = {-f for f in _forward_islands(mirrored, fps, policy)}
    return {f: REASON for f in sorted(forward | backward)}


def apply(
    rows: list[dict], fps: float | None, mode: str = DEFAULT
) -> tuple[list[dict], dict | None]:
    """Refuse island rows of one clip's automatic ball rows; off returns the input untouched.

    Only ``visible`` rows vote. A flagged row keeps its raw centre and becomes
    ``unsupported``, the same shape as an outside-image refusal, so nothing is
    deleted, moved or interpolated. A ``derived_estimate`` row whose nearest visible
    neighbours are both refused is an interpolation between wrong-object rows and is
    refused with them. Returns ``(rows, receipt)``; the receipt is ``None`` when off.
    """
    if validate_mode(mode) == DEFAULT:
        return rows, None
    if fps is None:
        raise ValueError("jump rejection requires the native cadence")
    policy = JumpPolicy(gap_step_seconds=GAP_STEP_SECONDS if mode == GAP_STEPS else 0.0)
    visible = [
        (int(r["frame"]), float(r["x1080"]), float(r["y1080"]))
        for r in rows
        if r.get("status") == "visible" and r.get("x1080") is not None
    ]
    flagged = island_flags(visible, float(fps), policy)
    frames = [f for f, _, _ in sorted(visible)]
    derived = set()
    for row in rows:
        if row.get("status") != "derived_estimate":
            continue
        frame = int(row["frame"])
        before = [f for f in frames if f < frame]
        after = [f for f in frames if f > frame]
        if before and after and before[-1] in flagged and after[0] in flagged:
            derived.add(frame)
    refused = []
    result = []
    for row in rows:
        frame = int(row["frame"])
        if (frame in flagged and row.get("status") == "visible") or frame in derived:
            refused.append(
                {
                    "frame": row["frame"],
                    "reason": REASON,
                    "original_status": row["status"],
                    "original_xy_native": [row.get("x1080"), row.get("y1080")],
                }
            )
            row = {
                **row,
                "status": "unsupported",
                "jump_rejection": REASON,
                "original_tracker_status": row["status"],
            }
        result.append(row)
    return result, {
        "schema": RECEIPT_SCHEMA,
        "mode": mode,
        "policy": dict(policy.__dict__),
        "labels_read": False,
        "raw_rows_retained": True,
        "refused_rows": len(refused),
        "refusals": refused,
    }
