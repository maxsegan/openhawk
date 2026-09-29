"""Build the automatic live-ball attempt ledger, one row per physical attempt.

Two assemblers live here.

``serve`` (the default) takes the serve-anchored attempts from ``serve_detector`` and the
decoded score path from ``score_grammar`` and writes one ledger row per attempt, carrying the
per-point score - server, set, games and points - as first-class columns.  Because the
attempt boundary is a detected serve rather than a gap in racket-impact audio, a first-serve
fault and the second serve that follows it are separate rows.

``legacy`` is the previous assembler: play-camera segments split at quiet gaps between audio
impact peaks, corroborated by a score or camera boundary.  It is kept behind ``--mode legacy``
so the old default remains reproducible.

``attempts`` splits an already-written *point* - a clip that the older segmentation cut once per
scored point, so that a fault or let and its replay live in the same clip - into one attempt per
serve.  It is behind ``--split-attempts``; with the flag off every point yields exactly one
attempt that keeps the point's own id, so the default output is unchanged.  See
``split_point_attempts`` for the rule and ``docs/wk1/solve_failures.md`` section 2.4 for why the
split is needed: a clip holding a fault plus its replay can never have one terminal flight, so
``reconstruction.complete_point_gate`` can never accept it.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from cv.pipeline.attempt_observation_scope import SCOPE_FIELDS as OBSERVATION_SCOPE_FIELDS


# SEG-001 measured this 3 s arm at 211 -> 103 splits with coverage held at 806/824;
# see cv/experiments/seg001_fix on lane-s1 (commit 3ae1f30).
QUIET_GAP_CORROBORATION_SECONDS = 3.0


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def score_transitions(rows: list[dict]) -> list[float]:
    fields = ("g1", "g2", "p1", "p2", "sets1", "sets2")
    output = []
    previous = None
    for row in rows:
        state = tuple(row.get(field, "") for field in fields)
        if previous is not None and state != previous:
            output.append(float(row["t_start"]))
        previous = state
    return output


def build_ledger(
    segments: list[dict],
    score_rows: list[dict],
    peaks: list[dict],
    *,
    peak_z: float,
    quiet_gap_seconds: float,
) -> tuple[list[dict], list[dict]]:
    transition_times = score_transitions(score_rows)
    peak_times = sorted(float(row["t"]) for row in peaks if float(row["z"]) >= peak_z)
    camera_boundary_times = sorted(
        {float(segment[field]) for segment in segments for field in ("t_start", "t_end")}
    )
    attempts = []
    held = []
    for segment_index, segment in enumerate(segments):
        start, end = float(segment["t_start"]), float(segment["t_end"])
        local_peaks = [time for time in peak_times if start <= time <= end]
        boundaries = [start]
        boundaries.extend(time for time in transition_times if start + 2.0 < time < end - 1.0)
        for left, right in zip(local_peaks, local_peaks[1:]):
            if right - left < quiet_gap_seconds:
                continue
            candidate = (left + right) / 2.0
            score_witness = any(
                abs(candidate - time) <= QUIET_GAP_CORROBORATION_SECONDS
                for time in transition_times
            )
            camera_witness = any(
                abs(candidate - time) <= QUIET_GAP_CORROBORATION_SECONDS
                for time in camera_boundary_times
            )
            if score_witness or camera_witness:
                boundaries.append(candidate)
        boundaries.append(end)
        ordered_boundaries = sorted(set(boundaries))
        boundaries = [ordered_boundaries[0]]
        for boundary in ordered_boundaries[1:]:
            if boundary - boundaries[-1] < 2.0 and boundary != end:
                boundaries[-1] = boundary
            else:
                boundaries.append(boundary)
        for part, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
            evidence = [time for time in local_peaks if left <= time <= right]
            if right - left < 2.0 or not evidence:
                held.append(
                    {
                        "segment": segment_index,
                        "part": part,
                        "t_start": left,
                        "t_end": right,
                        "reason": "no_impact_evidence" if not evidence else "too_short",
                    }
                )
                continue
            nearby_transition = next(
                (time for time in transition_times if left < time <= right + 5.0),
                None,
            )
            attempts.append(
                {
                    "rally_t_start": max(left, evidence[0] - 0.75),
                    "rally_t_end": right,
                    "segment_index": segment_index,
                    "segment_part": part,
                    "first_impact_t": evidence[0],
                    "impact_count": len(evidence),
                    "score_transition_t": nearby_transition,
                    "confidence": "high" if nearby_transition is not None else "provisional",
                    "reasons": "play_camera|impact_audio"
                    + ("|score_transition" if nearby_transition is not None else ""),
                }
            )
    attempts.sort(key=lambda row: row["rally_t_start"])
    for point, row in enumerate(attempts, start=1):
        row["pt"] = point
    return attempts, held


SCORE_COLUMNS = (
    "server",
    "sets_won_1",
    "sets_won_2",
    "set_number",
    "games_1",
    "games_2",
    "points_1",
    "points_2",
    "completed_sets",
)
LEDGER_FIELDS = [
    "pt",
    "rally_t_start",
    "rally_t_end",
    "segment_index",
    "segment_part",
    "first_impact_t",
    "impact_count",
    "score_transition_t",
    "confidence",
    "reasons",
    *SCORE_COLUMNS,
    "score_confidence",
    "score_source",
    "outcome_hint",
    "point_index",
    "attempt_role",
    "attempts_in_point",
]
POINT_FIELDS = [field for field in LEDGER_FIELDS if field != "attempt_role"]
SCORE_FOLLOW_SECONDS = 6.0
DEFAULT_MIN_SCORE_CONFIDENCE = 0.0
# Measured on the three finals: the decoded state is right 0.945 of the time when the crop
# behind it is under 15 s old and 0.2 of the time when it is over a minute old, so a state
# resting on a stale read is withheld rather than attached.
DEFAULT_MAX_READ_AGE_SECONDS = 15.0
# The board is redrawn a second or two after a point ends and the reader only samples once a
# second, so the state standing at the exact second of the serve can still be the previous
# point's.  The board cannot yet show *this* attempt's outcome while the rally is still being
# played, so reading the state a little way into the rally is safe and catches the update.
DEFAULT_ATTACH_LOOKAHEAD_SECONDS = 5.0


def score_state_at(score_rows: list[dict], moment: float) -> dict | None:
    """The decoded score in force at ``moment`` (the score the attempt is played on)."""
    chosen = None
    for row in score_rows:
        if float(row["t"]) <= moment:
            chosen = row
        else:
            break
    return chosen or (score_rows[0] if score_rows else None)


def score_advance_times(score_rows: list[dict]) -> list[float]:
    """Times at which the decoded score changed by a point."""
    times = []
    previous = None
    for row in score_rows:
        state = tuple(row[column] for column in SCORE_COLUMNS)
        if previous is not None and state != previous:
            times.append(float(row["t"]))
        previous = state
    return times


def score_row_confidence(row: dict | None) -> float:
    """The decoder's confidence in the state it attached, 1.0 for a v1 decode with none."""
    if row is None:
        return 0.0
    value = row.get("score_confidence")
    if value in (None, ""):
        return 1.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def score_row_age(row: dict | None) -> float:
    """How old the scoreboard read behind the attached state is, in seconds."""
    if row is None:
        return float("inf")
    value = row.get("read_age_seconds")
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return float("inf")


def build_scored_ledger(
    attempts: list[dict],
    peaks: list[dict],
    score_rows: list[dict],
    *,
    peak_z: float,
    min_confidence: float = DEFAULT_MIN_SCORE_CONFIDENCE,
    max_read_age_seconds: float = DEFAULT_MAX_READ_AGE_SECONDS,
    attach_lookahead_seconds: float = DEFAULT_ATTACH_LOOKAHEAD_SECONDS,
) -> list[dict]:
    """One ledger row per detected attempt, with the score it was played on attached.

    The owner's rule is that a wrong score is worse than no score, so an attempt whose
    decoded state the ensemble does not agree on, or whose nearest scoreboard read is too
    far away in time, carries empty score columns and ``score_source`` ``abstained``
    rather than the decoder's best guess.
    """
    peak_times = sorted(float(row["t"]) for row in peaks if float(row["z"]) >= peak_z)
    advances = score_advance_times(score_rows)
    rows = []
    for index, attempt in enumerate(sorted(attempts, key=lambda row: float(row["rally_t_start"]))):
        start = float(attempt["rally_t_start"])
        end = float(attempt["rally_t_end"])
        inside = [time for time in peak_times if start - 0.5 <= time <= end + 0.5]
        state = score_state_at(score_rows, max(start, min(start + attach_lookahead_seconds, end)))
        confidence = score_row_confidence(state)
        age = score_row_age(state)
        abstained = state is not None and (
            confidence < min_confidence or age > max_read_age_seconds
        )
        if abstained:
            state = None
        following = next(
            (time for time in advances if start < time <= end + SCORE_FOLLOW_SECONDS), None
        )
        reasons = ["serve_model", "play_view"]
        if inside:
            reasons.append("impact_audio")
        if state is not None:
            reasons.append("score_grammar")
        rows.append(
            {
                **{key: attempt[key] for key in OBSERVATION_SCOPE_FIELDS if key in attempt},
                "pt": index + 1,
                "rally_t_start": round(start, 3),
                "rally_t_end": round(end, 3),
                "segment_index": index,
                "segment_part": 0,
                "first_impact_t": round(inside[0], 3) if inside else round(start, 3),
                "impact_count": len(inside),
                "score_transition_t": round(following, 3) if following is not None else "",
                "confidence": "high" if following is not None else "provisional",
                "reasons": "|".join(reasons),
                **{
                    column: (state[column] if state is not None else "") for column in SCORE_COLUMNS
                },
                "score_confidence": round(confidence, 4) if state is not None else "",
                "score_source": (
                    "score_grammar_v1"
                    if state is not None
                    else ("abstained" if abstained else "none")
                ),
                "outcome_hint": "point" if following is not None else "fault_let_or_unscored",
            }
        )
    return rows


# --- serve witnesses: which attempts are the same score point -----------------------------
# A serve-anchored ledger emits one row per serve, so a first-serve fault and the second serve
# that follows it are two rows of one score point.  The reviewed 385-point Roland Garros
# ledger counts one window per score point, which is why 126 of 589 attempts there had no
# reviewed point to correspond to and 83 reviewed points held more than one attempt: both
# counts are dominated by second serves, not by wrong attempts.  Two witnesses that exist
# before any label is opened say whether an attempt opens a new point or continues the one
# before it:
#
# * the score decoder's state - a point award is the only legal transition between two
#   attempts of different points, so two attempts carrying the same decoded state (server,
#   sets, games, points) are the same point being served twice;
# * the clock - the second serve follows its fault inside the time the server needs to be
#   given a new ball, never a whole changeover later.
#
# Chosen on the twelve development windows, whose owner truth marks every attempt `point`,
# `serve_fault` or `let`: 15 s is the joint maximum of point precision (0.792) and point
# recall (0.785) over 6-45 s and unbounded, and it puts 75 of the 107 owner fault/let pairs
# in one point against a ceiling of 77.  See `cv.validation.s1_attempt_precision dev`.
POINT_CONTINUATION_MAX_GAP_SECONDS = 15.0


def _attempt_state(row: dict) -> tuple | None:
    """The decoded score a ledger row carries, or None when it abstained."""
    if not row.get("games_1") or not row.get("server"):
        return None
    return tuple(str(row.get(column, "")) for column in SCORE_COLUMNS)


def group_attempts_into_points(
    rows: list[dict],
    *,
    max_gap_seconds: float = POINT_CONTINUATION_MAX_GAP_SECONDS,
) -> list[list[int]]:
    """Index the ledger rows by the score point each attempt belongs to.

    Returns one list of row indices per point, in time order.  An attempt continues the
    point before it when the decoded score has not moved and the gap to the previous
    attempt's end is short enough to be a second serve; an attempt that abstains inherits
    the state standing over it, so a stretch with no readable board is not cut up.
    """
    ordered = sorted(range(len(rows)), key=lambda index: float(rows[index]["rally_t_start"]))
    points: list[list[int]] = []
    standing: tuple | None = None
    previous: dict | None = None
    for index in ordered:
        row = rows[index]
        state = _attempt_state(row)
        start = float(row["rally_t_start"])
        continues = previous is not None
        if continues and state is not None and standing is not None and state != standing:
            continues = False
        if continues and start - float(previous["rally_t_end"]) > max_gap_seconds:
            continues = False
        if continues:
            points[-1].append(index)
        else:
            points.append([index])
        if state is not None:
            standing = state
        previous = row
    return points


def annotate_points(
    rows: list[dict], *, max_gap_seconds: float = POINT_CONTINUATION_MAX_GAP_SECONDS
) -> list[dict]:
    """Add ``point_index``, ``attempt_role`` and ``attempts_in_point`` to every ledger row."""
    for number, members in enumerate(
        group_attempts_into_points(rows, max_gap_seconds=max_gap_seconds), start=1
    ):
        for position, index in enumerate(members):
            rows[index]["point_index"] = number
            rows[index]["attempt_role"] = "first_serve" if position == 0 else "continuation"
            rows[index]["attempts_in_point"] = len(members)
    return rows


def collapse_to_points(
    rows: list[dict], *, max_gap_seconds: float = POINT_CONTINUATION_MAX_GAP_SECONDS
) -> list[dict]:
    """One row per score point: the first serve's row, run on to the last attempt's end.

    The score, the server and the first impact are the first serve's, because that is when
    the point was played; the impact count and the confidence are the point's.
    """
    collapsed = []
    for number, members in enumerate(
        group_attempts_into_points(rows, max_gap_seconds=max_gap_seconds), start=1
    ):
        first, last = rows[members[0]], rows[members[-1]]
        row = dict(first)
        row["pt"] = number
        row["point_index"] = number
        row["attempts_in_point"] = len(members)
        row["rally_t_end"] = last["rally_t_end"]
        row["impact_count"] = sum(int(rows[index]["impact_count"] or 0) for index in members)
        transitions = [
            rows[index]["score_transition_t"]
            for index in members
            if rows[index].get("score_transition_t") not in (None, "")
        ]
        row["score_transition_t"] = transitions[-1] if transitions else ""
        row["confidence"] = "high" if transitions else "provisional"
        row["outcome_hint"] = "point" if transitions else "fault_let_or_unscored"
        row.pop("attempt_role", None)
        collapsed.append(row)
    return collapsed


# --- attempt split ------------------------------------------------------------------------
# ``reconstruction.MAX_SHOT_SECONDS``.  Two contacts further apart than this are never joined
# into one flight, and two contacts in different active-play spans are never joined either, so
# a point holding such a pair cannot be one attempt.  Kept as a literal rather than imported so
# that this module stays free of the reconstruction stack; the split test pins the two together.
DEFAULT_ATTEMPT_GAP_SECONDS = 3.5
ATTEMPT_FIELDS = [
    "attempt_id",
    "parent_point",
    "match_id",
    "attempt_index",
    "attempt_count",
    "fps",
    "start_frame",
    "end_frame",
    "active_spans",
    "contact_frames",
    "contacts",
    "split_reasons",
]


def _contacts_in(frames: list[float], start: float, end: float) -> list[float]:
    return [frame for frame in frames if start <= frame <= end]


def split_point_attempts(
    point_id: str,
    active_spans: list[list[float]],
    contact_frames: list[float],
    *,
    fps: float,
    gap_seconds: float = DEFAULT_ATTEMPT_GAP_SECONDS,
    enabled: bool = False,
) -> list[dict]:
    """Split one point clip into attempts: a serve and the rally it starts, until the ball is dead.

    Label-free.  The only evidence used is the point's active-play spans and its automatic
    contact emissions, both of which exist before any truth is opened.

    Two cuts are made:

    * between active-play spans, because the fitter never joins contacts across a span; and
    * inside a span, midway between two consecutive contacts more than ``gap_seconds`` apart,
      because no flight can span that gap either.

    A fragment carrying no contact is dead time and is dropped rather than emitted as an
    attempt.  With ``enabled`` false, or when the rule finds no cut, the point yields exactly
    one attempt that keeps the point's own id, so nothing downstream changes.
    """
    spans = [[float(start), float(end)] for start, end in active_spans]
    frames = sorted(float(frame) for frame in contact_frames)
    whole = [
        {
            "attempt_id": point_id,
            "parent_point": point_id,
            "attempt_index": 1,
            "active_spans": spans,
            "contact_frames": list(frames),
            "split_reasons": [],
        }
    ]
    if not enabled or not spans or not frames:
        return _finalize_attempts(whole, fps=fps)
    limit = gap_seconds * float(fps)
    fragments: list[dict] = []
    for span_index, (start, end) in enumerate(spans):
        inside = _contacts_in(frames, start, end)
        if not inside:
            continue
        cuts = [start]
        reasons = ["span_boundary"] if span_index else []
        for left, right in zip(inside, inside[1:]):
            if right - left > limit:
                cuts.append((left + right) / 2.0)
        cuts.append(end)
        for part, (left, right) in enumerate(zip(cuts, cuts[1:])):
            piece = _contacts_in(frames, left, right)
            if not piece:
                continue
            fragment_reasons = list(reasons) if part == 0 else ["contact_gap"]
            fragments.append(
                {
                    "active_spans": [[left, right]],
                    "contact_frames": piece,
                    "split_reasons": fragment_reasons,
                }
            )
    if len(fragments) <= 1:
        return _finalize_attempts(whole, fps=fps)
    attempts = []
    for index, fragment in enumerate(fragments, start=1):
        attempts.append(
            {
                "attempt_id": f"{point_id}__a{index:02d}",
                "parent_point": point_id,
                "attempt_index": index,
                **fragment,
            }
        )
    return _finalize_attempts(attempts, fps=fps)


def _finalize_attempts(attempts: list[dict], *, fps: float) -> list[dict]:
    for attempt in attempts:
        spans = attempt["active_spans"]
        attempt["attempt_count"] = len(attempts)
        attempt["fps"] = float(fps)
        attempt["match_id"] = attempt["parent_point"].split("__")[0]
        attempt["start_frame"] = min((span[0] for span in spans), default=0.0)
        attempt["end_frame"] = max((span[1] for span in spans), default=0.0)
        attempt["contacts"] = len(attempt["contact_frames"])
    return attempts


def build_attempt_ledger(
    active_play: dict,
    emissions: list[dict],
    *,
    split_attempts: bool,
    gap_seconds: float = DEFAULT_ATTEMPT_GAP_SECONDS,
) -> list[dict]:
    """One row per attempt over a whole cohort of point clips.

    ``active_play`` is an ``active_play_v*.json`` payload keyed ``match_id/clip``; ``emissions``
    are the automatic event rows the runner hands Stage 6.
    """
    contacts: dict[str, list[float]] = {}
    for row in emissions:
        if row.get("event_type") != "contact" or row.get("abstain") is True:
            continue
        contacts.setdefault(str(row["clip"]), []).append(float(row["frame"]))
    rows = []
    for key in sorted(active_play):
        match_id, _, clip = key.partition("/")
        point_id = f"{match_id}__{clip}"
        entry = active_play[key]
        rows.extend(
            split_point_attempts(
                point_id,
                entry.get("active_spans", []),
                contacts.get(point_id, []),
                fps=float(entry["fps"]),
                gap_seconds=gap_seconds,
                enabled=split_attempts,
            )
        )
    return rows


def write_attempt_ledger(output: Path, rows: list[dict]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ATTEMPT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "active_spans": ";".join(
                        f"{span[0]:g}:{span[1]:g}" for span in row["active_spans"]
                    ),
                    "contact_frames": ";".join(f"{frame:g}" for frame in row["contact_frames"]),
                    "split_reasons": "|".join(row["split_reasons"]),
                }
            )


def write_ledger(output: Path, rows: list[dict], *, fields: list[str] | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(fields or LEDGER_FIELDS)
            + [
                key
                for key in OBSERVATION_SCOPE_FIELDS
                if any(key in row for row in rows) and key not in (fields or LEDGER_FIELDS)
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("serve", "legacy", "attempts"), default="serve")
    parser.add_argument("--attempts", type=Path, help="serve_detector attempts (serve mode)")
    parser.add_argument("--score-state", type=Path, help="score_grammar decode (serve mode)")
    parser.add_argument("--segments", type=Path, help="play-camera segments (legacy mode)")
    parser.add_argument("--score-runs", type=Path, help="raw score runs (legacy mode)")
    parser.add_argument("--active-play", type=Path, help="active_play_v*.json (attempts mode)")
    parser.add_argument("--emissions", type=Path, help="automatic event rows (attempts mode)")
    parser.add_argument(
        "--split-attempts",
        action="store_true",
        help="split a point that holds more than one serve into one attempt each",
    )
    parser.add_argument(
        "--attempt-gap-seconds",
        type=float,
        default=DEFAULT_ATTEMPT_GAP_SECONDS,
        help="contacts further apart than this start a new attempt (attempts mode)",
    )
    parser.add_argument(
        "--rows",
        choices=("attempts", "points"),
        default="attempts",
        help="one row per detected serve, or one row per score point (serve mode)",
    )
    parser.add_argument(
        "--point-continuation-gap",
        type=float,
        default=POINT_CONTINUATION_MAX_GAP_SECONDS,
        help="a serve this long after the previous attempt opens a new point (serve mode)",
    )
    parser.add_argument("--serve-peaks", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--peak-z", type=float, default=5.0)
    parser.add_argument("--quiet-gap-seconds", type=float, default=6.0)
    parser.add_argument(
        "--min-score-confidence",
        type=float,
        default=DEFAULT_MIN_SCORE_CONFIDENCE,
        help="attach no score below this decoder confidence (serve mode)",
    )
    parser.add_argument(
        "--score-attach-lookahead",
        type=float,
        default=DEFAULT_ATTACH_LOOKAHEAD_SECONDS,
        help="read the score this many seconds into the rally, clipped to its end",
    )
    parser.add_argument(
        "--max-score-read-age",
        type=float,
        default=DEFAULT_MAX_READ_AGE_SECONDS,
        help="attach no score when the nearest scoreboard read is older (serve mode)",
    )
    args = parser.parse_args()
    held: list[dict] = []
    if args.mode == "attempts":
        if args.active_play is None or args.emissions is None:
            parser.error("--active-play and --emissions are required in attempts mode")
        rows = build_attempt_ledger(
            json.loads(args.active_play.read_text()),
            json.loads(args.emissions.read_text()),
            split_attempts=args.split_attempts,
            gap_seconds=args.attempt_gap_seconds,
        )
        write_attempt_ledger(args.output, rows)
        parents = {row["parent_point"] for row in rows}
        args.report.write_text(
            json.dumps(
                {
                    "schema": "automatic_attempt_ledger_v1",
                    "mode": args.mode,
                    "labels_loaded": False,
                    "split_attempts": bool(args.split_attempts),
                    "attempt_gap_seconds": args.attempt_gap_seconds,
                    "points": len(parents),
                    "attempts": len(rows),
                    "split_points": len(parents)
                    - sum(1 for row in rows if row["attempt_count"] == 1),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return
    if args.serve_peaks is None:
        parser.error("--serve-peaks is required in serve and legacy modes")
    if args.mode == "serve":
        if args.attempts is None:
            parser.error("--attempts is required in serve mode")
        score_rows = read_rows(args.score_state) if args.score_state else []
        attempts = build_scored_ledger(
            read_rows(args.attempts),
            read_rows(args.serve_peaks),
            score_rows,
            peak_z=args.peak_z,
            min_confidence=args.min_score_confidence,
            max_read_age_seconds=args.max_score_read_age,
            attach_lookahead_seconds=args.score_attach_lookahead,
        )
        annotate_points(attempts, max_gap_seconds=args.point_continuation_gap)
        if args.rows == "points":
            attempts = collapse_to_points(attempts, max_gap_seconds=args.point_continuation_gap)
    else:
        if args.segments is None or args.score_runs is None:
            parser.error("--segments and --score-runs are required in legacy mode")
        attempts, held = build_ledger(
            read_rows(args.segments),
            read_rows(args.score_runs),
            read_rows(args.serve_peaks),
            peak_z=args.peak_z,
            quiet_gap_seconds=args.quiet_gap_seconds,
        )
    write_ledger(
        args.output,
        attempts,
        fields=POINT_FIELDS if args.mode == "serve" and args.rows == "points" else LEDGER_FIELDS,
    )
    args.report.write_text(
        json.dumps(
            {
                "schema": "automatic_point_ledger_v1",
                "mode": args.mode,
                "rows": args.rows,
                "point_continuation_gap_seconds": args.point_continuation_gap,
                "labels_loaded": False,
                "attempts": len(attempts),
                "score_points": len(
                    {row["point_index"] for row in attempts if row.get("point_index")}
                )
                or None,
                # A serve-anchored stage emits one row per serve, so the ledger's own count of
                # score points is what a score-keeping consumer needs; the attempt row stays the
                # unit every downstream clip is cut for.
                "continuation_attempts": sum(
                    row.get("attempt_role") == "continuation" for row in attempts
                ),
                "multi_attempt_points": len(
                    {
                        row.get("point_index")
                        for row in attempts
                        if int(row.get("attempts_in_point") or 1) > 1
                    }
                ),
                "high_confidence": sum(row["confidence"] == "high" for row in attempts),
                "provisional": sum(row["confidence"] == "provisional" for row in attempts),
                "with_score_columns": sum(
                    bool(row.get("score_source", "none") not in ("none", "abstained"))
                    for row in attempts
                ),
                "score_abstentions": sum(
                    row.get("score_source") == "abstained" for row in attempts
                ),
                "held_intervals": held,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
