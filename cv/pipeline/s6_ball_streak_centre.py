"""Label-free streak-centre correction of automatic ball centres.

On a fast ball the detector centre sits near the leading end of the motion-blur streak,
while the ball's position at the frame's timestamp (and the human click) is its middle.
Measured on 46 development attempts (matched_fresh36_v1 + matched_fresh12_v1, LL vs AL,
same native frame, ``cv/experiments/streak_centre/measure_offset.py``): the automatic
centre is ahead along travel in 46 of 46 attempts, by a median 0.2 frame of travel at
15-50 px/frame (4.8 px at 20-25 px/frame, 8.8 px at 40-50), and not sideways.

The correction moves each automatic centre back along its own track velocity by
``GAIN_FRAMES`` frames of travel. The velocity is the step from the previous native
frame's row (the direction the blur trails), else the step to the next one; a row with
no adjacent row is left in place. One constant, set once on the development pairs and
frozen; nothing is fitted per clip. Default off: ``apply`` is a no-op unless the shared
policy key ``automatic_ball_streak_centre`` is declared "on" (absent and "off" are
identical). The raw centre is kept on the row under ``streak_centre``.
"""

from __future__ import annotations

import math

FIELD = "automatic_ball_streak_centre"
DEFAULT = "off"
MODES = (DEFAULT, "on")
#: frames of image travel between the detector centre and the streak middle (frozen
#: 2026-09-28 from OFFSET_MEASURE.json: backward velocity, k=0.2 minimises the mean
#: error above 15 px/frame; 0.15-0.25 are within 0.2 px of it)
GAIN_FRAMES = 0.2
SHIFTED_STATUSES = ("visible", "derived_estimate")
RECEIPT_SCHEMA = "automatic_ball_streak_centre_v1"


def validate_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError("automatic_ball_streak_centre must be 'off' or 'on'")
    return mode


def apply(rows: list[dict], mode: str = DEFAULT) -> tuple[list[dict], dict | None]:
    """Shift one clip's automatic centres back along travel; off returns the input untouched.

    Returns ``(rows, receipt)``; the receipt is ``None`` when off. Rows that are not
    ``visible`` or ``derived_estimate`` are copied unchanged and never lend a velocity.
    """

    if validate_mode(mode) == DEFAULT:
        return rows, None
    track = {
        int(r["frame"]): (float(r["x1080"]), float(r["y1080"]))
        for r in rows
        if r.get("status") in SHIFTED_STATUSES and r.get("x1080") is not None
    }
    result = []
    shifts = []
    for row in rows:
        frame = int(row["frame"])
        here = track.get(frame) if row.get("status") in SHIFTED_STATUSES else None
        if here is None:
            result.append(row)
            continue
        before, after = track.get(frame - 1), track.get(frame + 1)
        if before is not None:
            velocity, basis = (here[0] - before[0], here[1] - before[1]), "previous_frame"
        elif after is not None:
            velocity, basis = (after[0] - here[0], after[1] - here[1]), "next_frame"
        else:
            result.append(row)
            continue
        dx, dy = -GAIN_FRAMES * velocity[0], -GAIN_FRAMES * velocity[1]
        shifts.append(math.hypot(dx, dy))
        result.append(
            {
                **row,
                "x1080": here[0] + dx,
                "y1080": here[1] + dy,
                "streak_centre": {
                    "raw_xy": [here[0], here[1]],
                    "shift_px": [dx, dy],
                    "velocity_basis": basis,
                    "gain_frames": GAIN_FRAMES,
                },
            }
        )
    shifts.sort()
    return result, {
        "schema": RECEIPT_SCHEMA,
        "gain_frames": GAIN_FRAMES,
        "shifted_rows": len(shifts),
        "median_shift_px": shifts[len(shifts) // 2] if shifts else 0.0,
        "max_shift_px": shifts[-1] if shifts else 0.0,
    }
