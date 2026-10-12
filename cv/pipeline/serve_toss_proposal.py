"""Propose a serve contact from the automatic ball track: a toss, then a departure.

A serve is the one stroke with no incoming flight. The server tosses the ball up from
the hand, it peaks above the head and starts to fall, and the racket meets it near the
top. The event decoder sometimes never places a row there, or abstains on it. This
module reads only the automatic ball track and the automatic player boxes and returns
candidate serve contacts. It decides nothing: a proposal is a candidate the caller
must still put through the existing gates and the fitter.

A proposal needs every one of these, with lengths in the server box's own height so
the same rule holds near and far and at any zoom:

* a rise of at least ``MIN_RISE_BOX_HEIGHTS`` lasting at least ``MIN_RISE_SECONDS``
  that starts inside the server box (grown by ``BOX_MARGIN``) and is more vertical
  than sideways in the picture;
* an apex at or above the top of the server box and horizontally over it;
* at least ``MIN_DESCENT_SAMPLES`` samples after the apex that each land within
  ``TOSS_RESIDUAL_BOX_HEIGHTS`` of the toss parabola fitted to the samples before
  them (x linear and y quadratic in time; the fit uses at most the last
  ``TOSS_FIT_SECONDS`` of the rise, the ballistic part);
* a departure: over the next ``OUTGOING_SECONDS`` every sample is off the toss
  parabola by more than ``DEPARTURE_BOX_HEIGHTS`` and further than the one before,
  the samples lie on a line (RMS at most ``OUTGOING_RMS_BOX_HEIGHTS``) heading to the
  other end of the court (down the picture for the far server, up it for the near
  one), at least ``MIN_SPEED_RATIO`` times as fast as the falling toss.

The contact epoch is where the toss parabola and the outgoing line are closest,
searched between the last toss sample and the first departed one.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

POLICY_KEY = "serve_toss_proposal"
SCHEMA = "serve_toss_proposal_v1"
SOURCE = "serve_toss_proposal"

# Units are server-box heights and seconds.
MIN_RISE_BOX_HEIGHTS = 0.35
MIN_RISE_SECONDS = 0.12
MAX_TOSS_SECONDS = 1.6
TOSS_FIT_SECONDS = 0.30
BOX_MARGIN = 0.5
APEX_ABOVE_TOP_BOX_HEIGHTS = 0.0
APEX_NOISE_BOX_HEIGHTS = 0.02
MIN_DESCENT_SAMPLES = 2
TOSS_RESIDUAL_BOX_HEIGHTS = 0.05
MIN_RESIDUAL_PX = 3.0
DEPARTURE_BOX_HEIGHTS = 0.10
OUTGOING_SECONDS = 0.15
MIN_OUTGOING_SAMPLES = 3
OUTGOING_RMS_BOX_HEIGHTS = 0.05
MAX_GAP_SECONDS = 0.10
MIN_SPEED_RATIO = 2.0
UNSEEN_RISE_SECONDS = 0.25
MIN_DESCENT_SECONDS = 0.06


@dataclass(frozen=True)
class ServeProposal:
    frame: float
    server: str
    apex_frame: int
    toss_start_frame: int
    last_toss_frame: int
    first_out_frame: int
    box_height_px: float
    rise_box_heights: float
    apex_above_top_box_heights: float
    departure_box_heights: float
    outgoing_rms_box_heights: float
    toss_speed_px_s: float
    outgoing_speed_px_s: float
    image_x: float
    image_y: float

    def as_dict(self) -> dict:
        return {"schema": SCHEMA, **asdict(self)}


def _box_at(boxes: dict[int, dict[str, tuple]], frame: int, side: str, fps: float):
    reach = max(1, round(0.2 * fps))
    for offset in range(reach + 1):
        for candidate in (frame - offset, frame + offset):
            box = (boxes.get(candidate) or {}).get(side)
            if box is not None:
                return box
    return None


def _runs(frames: list[int], fps: float) -> list[list[int]]:
    """Split sorted frames where the track has a gap longer than MAX_GAP_SECONDS."""

    runs: list[list[int]] = []
    for frame in frames:
        if runs and (frame - runs[-1][-1]) / fps <= MAX_GAP_SECONDS:
            runs[-1].append(frame)
        else:
            runs.append([frame])
    return runs


def _parabola(track, frames: list[int]) -> np.ndarray:
    t = np.asarray(frames, dtype=float)
    xy = np.stack([track[f] for f in frames])
    return np.concatenate([np.polyfit(t, xy[:, 0], 1), np.polyfit(t, xy[:, 1], 2)])


def _toss_at(coef: np.ndarray, t: float) -> np.ndarray:
    return np.asarray([np.polyval(coef[:2], t), np.polyval(coef[2:], t)])


def _is_apex(track, run: list[int], index: int, tolerance: float) -> bool:
    y = track[run[index]][1]
    before = track[run[index - 1]][1]
    after = track[run[index + 1]][1]
    return before >= y - tolerance and after >= y and (before > y or after > y)


def propose(
    track: dict[int, np.ndarray],
    boxes: dict[int, dict[str, tuple]],
    fps: float,
) -> list[ServeProposal]:
    """Every toss-and-departure in one clip, in frame order."""

    return [found for _apex, found, _why in _scan(track, boxes, fps) if found is not None]


def diagnose(
    track: dict[int, np.ndarray], boxes: dict[int, dict[str, tuple]], fps: float, frame: float
) -> list[tuple[int, str]]:
    """Why each apex in the toss window before ``frame`` did or did not propose."""

    return [
        (apex, f"proposal@{found.frame}" if found else why)
        for apex, found, why in _scan(track, boxes, fps)
        if frame - MAX_TOSS_SECONDS * fps <= apex <= frame + 2
    ]


def _scan(track, boxes, fps: float):
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("serve proposal needs a positive fps")
    out = []
    for run in _runs(sorted(track), fps):
        index = 1
        while index < len(run) - 1:
            if not _is_apex(track, run, index, 2.0):
                index += 1
                continue
            found, why = _from_apex(track, boxes, fps, run, index)
            out.append((run[index], found, why))
            if found is not None:
                index = run.index(found.first_out_frame)
            else:
                index += 1
    return out


def _server(track, boxes, fps: float, run: list[int], index: int):
    """The player the toss rose over, and where the toss starts.

    The rise is either seen from inside the server box, or not tracked at all: the
    run then starts within ``UNSEEN_RISE_SECONDS`` of the apex.
    """

    apex = run[index]
    apex_xy = track[apex]
    best = None
    why = []
    for side in ("near", "far"):
        box = _box_at(boxes, apex, side, fps)
        if box is None:
            why.append(f"{side}:no_box")
            continue
        x0, y0, x1, y1 = box
        height = y1 - y0
        if height <= 0:
            continue
        margin = BOX_MARGIN * height

        def near_server(xy) -> bool:
            return bool(x0 - margin <= xy[0] <= x1 + margin and y0 - margin <= xy[1] <= y1 + margin)

        if not x0 - margin <= apex_xy[0] <= x1 + margin:
            why.append(f"{side}:apex_not_over_box")
            continue
        above = (y0 - apex_xy[1]) / height
        if above < APEX_ABOVE_TOP_BOX_HEIGHTS:
            why.append(f"{side}:apex_below_box_top")
            continue
        tolerance = APEX_NOISE_BOX_HEIGHTS * height + 1.0
        start = index
        while (
            start > 0
            and track[run[start - 1]][1] >= track[run[start]][1] - tolerance
            and (apex - run[start - 1]) / fps <= MAX_TOSS_SECONDS
        ):
            start -= 1
        inside = [k for k in range(start, index) if near_server(track[run[k]])]
        rise = 0.0
        origin_index = None
        if inside:
            origin_index = max(inside, key=lambda k: track[run[k]][1])
            origin = track[run[origin_index]]
            rise = (origin[1] - apex_xy[1]) / height
            if (
                (apex - run[origin_index]) / fps < MIN_RISE_SECONDS
                or rise < MIN_RISE_BOX_HEIGHTS
                or abs(origin[0] - apex_xy[0]) > origin[1] - apex_xy[1]
            ):
                origin_index = None
        if origin_index is None:
            if start != 0 or (apex - run[0]) / fps > UNSEEN_RISE_SECONDS:
                why.append(f"{side}:no_rise_from_server")
                continue
            origin_index = 0
            rise = 0.0
        if best is None or rise > best["rise"]:
            best = {
                "side": side,
                "height": height,
                "rise": rise,
                "above": above,
                "origin_index": origin_index,
            }
    return best, ",".join(why)


def _from_apex(track, boxes, fps: float, run: list[int], index: int):
    server, why = _server(track, boxes, fps, run, index)
    if server is None:
        return None, "no_toss:" + why
    height = server["height"]
    apex = run[index]
    first = max(server["origin_index"], 0)
    while (apex - run[first]) / fps > TOSS_FIT_SECONDS and first < index - 2:
        first += 1
    toss = run[first : index + 1]
    if len(toss) < 3:
        return None, "toss_too_few_samples"
    tolerance = max(MIN_RESIDUAL_PX, TOSS_RESIDUAL_BOX_HEIGHTS * height)
    last = index
    while last + 1 < len(run):
        coef = _parabola(track, toss)
        following = run[last + 1]
        if np.linalg.norm(track[following] - _toss_at(coef, float(following))) > tolerance:
            break
        toss = toss + [following]
        last += 1
    if last - index < MIN_DESCENT_SAMPLES or (run[last] - apex) / fps < MIN_DESCENT_SECONDS:
        return None, "toss_not_falling_on_its_parabola"
    coef = _parabola(track, toss)
    first_out = last + 1
    if first_out >= len(run):
        return None, "track_ends_on_the_toss"
    horizon = run[last] + OUTGOING_SECONDS * fps
    outgoing = [f for f in run[first_out:] if f <= horizon]
    if len(outgoing) < MIN_OUTGOING_SAMPLES:
        outgoing = run[first_out : first_out + MIN_OUTGOING_SAMPLES]
    if len(outgoing) < MIN_OUTGOING_SAMPLES:
        return None, "too_few_outgoing_samples"
    residuals = [
        float(np.linalg.norm(track[f] - _toss_at(coef, float(f))) / height) for f in outgoing
    ]
    if residuals[-1] < DEPARTURE_BOX_HEIGHTS or any(
        b < a for a, b in zip(residuals, residuals[1:], strict=False)
    ):
        return None, "not_leaving_the_toss"
    t = np.asarray(outgoing, dtype=float)
    xy = np.stack([track[f] for f in outgoing])
    line = [np.polyfit(t, xy[:, k], 1) for k in (0, 1)]
    fitted = np.stack([np.polyval(line[0], t), np.polyval(line[1], t)], axis=1)
    rms = float(np.sqrt(np.mean(np.sum((fitted - xy) ** 2, axis=1)))) / height
    if rms > OUTGOING_RMS_BOX_HEIGHTS:
        return None, "outgoing_not_a_line"
    velocity = np.asarray([line[0][0], line[1][0]])
    if not (velocity[1] > 0 if server["side"] == "far" else velocity[1] < 0):
        return None, "outgoing_not_toward_the_other_end"
    out_speed = float(np.linalg.norm(velocity)) * fps
    toss_velocity = np.asarray([coef[0], 2 * coef[2] * run[last] + coef[3]], dtype=float)
    toss_speed = float(np.linalg.norm(toss_velocity)) * fps
    if out_speed < MIN_SPEED_RATIO * toss_speed:
        return None, "outgoing_not_faster_than_the_toss"
    grid = np.linspace(float(run[last]), float(run[first_out]), 41)
    gaps = [
        np.linalg.norm(
            _toss_at(coef, g) - np.asarray([np.polyval(line[0], g), np.polyval(line[1], g)])
        )
        for g in grid
    ]
    epoch = float(grid[int(np.argmin(gaps))])
    point = _toss_at(coef, epoch)
    return ServeProposal(
        frame=round(epoch, 3),
        server=server["side"],
        apex_frame=int(apex),
        toss_start_frame=int(run[server["origin_index"]]),
        last_toss_frame=int(run[last]),
        first_out_frame=int(run[first_out]),
        box_height_px=round(float(height), 1),
        rise_box_heights=round(float(server["rise"]), 3),
        apex_above_top_box_heights=round(float(server["above"]), 3),
        departure_box_heights=round(residuals[0], 3),
        outgoing_rms_box_heights=round(rms, 4),
        toss_speed_px_s=round(toss_speed, 1),
        outgoing_speed_px_s=round(out_speed, 1),
        image_x=round(float(point[0]), 2),
        image_y=round(float(point[1]), 2),
    ), "proposal"


# Admission into the decoder's emission rows. A serve is the first contact of its
# point and is struck overhead, so a proposal is admitted only when all of these hold:
# no model-accepted contact of the clip lies more than ``FIRST_CONTACT_SECONDS`` before
# it, none lies within ``FIRST_CONTACT_SECONDS`` of it (the decoder already has the
# serve), no decoder bounce row sits at the departure, and the contact point is within
# ``CONTACT_ABOVE_TOP_BOX_HEIGHTS`` above to ``CONTACT_BELOW_TOP_BOX_HEIGHTS`` below the
# top of the server box. An abstained decoder contact at the proposal is released with
# the decoder's own epoch; otherwise a new row is added at the proposal epoch.
FIRST_CONTACT_SECONDS = 0.25
BOUNCE_AT_DEPARTURE_SECONDS = 0.10
SAME_ROW_SECONDS = 0.10
CONTACT_ABOVE_TOP_BOX_HEIGHTS = 0.60
CONTACT_BELOW_TOP_BOX_HEIGHTS = 0.25


def _same_clip(row: dict, match_id: str, clip: str) -> bool:
    if row.get("match_id") != match_id:
        return False
    name = str(row.get("clip", ""))
    return name == clip or name.endswith("__" + clip)


def admit_serve_proposals(
    rows: list[dict],
    *,
    match_id: str,
    clip: str,
    track: dict[int, np.ndarray],
    boxes_by_frame: dict[int, list[tuple[str, tuple[float, float, float, float]]]],
    fps: float,
) -> tuple[list[dict], dict]:
    """Add or release the serve contacts the decoder did not accept."""

    census: dict = {"schema": SCHEMA, "proposed": 0, "added": 0, "released": 0, "refused": []}
    boxes = {
        frame: {side: tuple(map(float, box)) for side, box in entries}
        for frame, entries in boxes_by_frame.items()
    }
    proposals = propose(track, boxes, fps)
    census["proposed"] = len(proposals)
    if not proposals:
        return rows, census
    mine = [row for row in rows if _same_clip(row, match_id, clip)]
    accepted = sorted(
        float(row["frame"])
        for row in mine
        if row.get("event_type") == "contact" and row.get("abstain") is False
    )
    bounces = [float(row["frame"]) for row in mine if row.get("event_type") == "bounce"]
    near = FIRST_CONTACT_SECONDS * fps
    output = list(rows)
    for proposal in proposals:
        reason = None
        if any(frame < proposal.frame - near for frame in accepted):
            reason = "after_an_accepted_contact"
        elif any(abs(frame - proposal.frame) <= near for frame in accepted):
            reason = "decoder_accepted_contact_here"
        elif any(
            abs(frame - proposal.frame) <= BOUNCE_AT_DEPARTURE_SECONDS * fps for frame in bounces
        ):
            reason = "departure_is_a_decoder_bounce"
        else:
            box = _box_at(boxes, int(round(proposal.frame)), proposal.server, fps)
            height = box[3] - box[1]
            level = (proposal.image_y - box[1]) / height
            if not -CONTACT_ABOVE_TOP_BOX_HEIGHTS <= level <= CONTACT_BELOW_TOP_BOX_HEIGHTS:
                reason = "contact_not_at_racket_height"
        if reason is not None:
            census["refused"].append({"frame": proposal.frame, "reason": reason})
            continue
        evidence = {**proposal.as_dict(), "certifies_physical_ending": False}
        same = [
            (index, row)
            for index, row in enumerate(output)
            if _same_clip(row, match_id, clip)
            and row.get("event_type") == "contact"
            and row.get("abstain") is True
            and abs(float(row["frame"]) - proposal.frame) <= max(2.0, SAME_ROW_SECONDS * fps)
        ]
        if same:
            index, row = min(same, key=lambda item: abs(float(item[1]["frame"]) - proposal.frame))
            output[index] = {
                **row,
                "abstain": False,
                "gate_held": False,
                "point_gate_verdict": "retain",
                "point_gate_failure_reasons": [],
                POLICY_KEY: {**evidence, "action": "released_decoder_row"},
            }
            census["released"] += 1
        else:
            output.append(
                {
                    "match_id": match_id,
                    "clip": f"{match_id}__{clip}",
                    "event_type": "contact",
                    "frame": float(round(proposal.frame)),
                    "abstain": False,
                    "model_abstain": None,
                    "gate_held": False,
                    "point_gate_verdict": "retain",
                    "point_gate_failure_reasons": [],
                    "source": SOURCE,
                    "location": {
                        "frame_subpixel": proposal.frame,
                        "image_x": proposal.image_x,
                        "image_y": proposal.image_y,
                        "image_coordinate_space": "native_1920x1080",
                        "source": SOURCE,
                    },
                    POLICY_KEY: {**evidence, "action": "added_row"},
                }
            )
            census["added"] += 1
        census.setdefault("admitted", []).append(proposal.frame)
    return output, census
