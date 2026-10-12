"""Who is near, who is far, who serves and who hits each shot, per attempt (owner 2026-10-05).

Identity is required to use the data, so it is written into the match outputs
(``PLAYER_IDENTITY.json``, ``MATCH_RESULT.json`` attempts and flights, ``OUTCOMES.json``), the
portal documents and the pose layers, each claim with its evidence and a confidence. Usage on a
finished match directory, or read-only over published matches::

    python -m cv.pipeline.player_names --out OUT --match-id MATCH [--portal OUT/portal/3d]
    python -m cv.pipeline.player_names --pool ARCHIVE_RUN_DIR ... --report pooled.json

The decision is one match-level assignment from automatic or public witnesses:

1. SCOREBOARD ROW -> NAME. The match metadata gives winner, loser and final score (winner
   first). The decoded board scores in the point ledger are row 1 first, so the completed sets,
   and the games reached inside each set, fit only one row order. The metadata is public match
   metadata, not an evaluation label.
2. CHANGEOVER RULES. Ends change after game 1 and then after every odd game total in a set,
   at a set end when that set's game count is odd, and every 6 points in a tiebreak
   (``player_identity.changeovers_before``). From each scored attempt's decoded score this gives
   the absolute number of end changes, so which row is near is fixed for the whole match up to
   one bit: which row started near.
3. SERVE SIDES. Each attempt whose fit saw the serve on a side, with its decoded server row,
   votes that bit. Rule-vs-evidence disagreements are counted; they expose score or side errors.

Per attempt the identity is then ``direct`` (its own serve side and server row agree with the
match assignment), ``propagated`` (no own serve observation: the changeover rules place it from
its own score, its point's score, or scored neighbours with the same end-change count; an
attempt whose own serve side contradicts the rules is also named by the rules, flagged as a
disagreement at half confidence), or ``unresolved`` (null: no score places it, or its whole
set's serve sides contradict the rules).

``TENNIS_PORTAL_PLAYER_NAMES=off`` skips the pass (rollback); the viewer then shows near/far.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

from cv.pipeline.player_identity import (
    ASSUMED_TIEBREAK_POINTS,
    PARTICLES,
    TIEBREAK_GAMES,
    changeovers_before,
    roster_from_match_id,
)

SCHEMA = "player_names_v1"
FILE = "PLAYER_IDENTITY.json"
ENV = "TENNIS_PORTAL_PLAYER_NAMES"
METADATA = Path("match_metadata/matches.csv")
LEDGER = "automatic_point_ledger_v1.csv"
NET_Y_M = 23.77 / 2
LABEL_MAX_LETTERS = 8
# The start bit is a whole-match vote; a handful of agreeing serves is a measurement, and wide
# disagreement means the score decode or the serve sides cannot be trusted together.
MIN_SIDE_VOTES = 3
MIN_SIDE_AGREEMENT = 0.8
# Each set is checked against the match bit on its own: a set whose serves mostly disagree
# means an end-change count is off inside the match (a misread set or tiebreak), and naming
# that set by the rules would swap every name in it.
MIN_SET_VOTES = 3
MIN_SET_AGREEMENT = 0.75
# Board games that exceed the metadata's set score under one row order rule that order out.
# A few decoder misreads are tolerated, but the losing order must be clearly worse.
MIN_ROW_MARGIN = 3
MAX_ROW_VIOLATION_SHARE = 0.05
SOURCES = ("direct", "propagated", "unresolved")


def enabled(environ=None) -> bool:
    return (environ or os.environ).get(ENV, "on") != "off"


# --------------------------------------------------------------------------- labels


def _letters(text: str) -> int:
    return sum(ch.isalpha() for ch in text)


def _cap(word: str) -> str:
    return word[:1].upper() + word[1:]


def surname(full_name: str) -> str:
    """The family name of a metadata full name ("Alex de Minaur" -> "de Minaur").

    Metadata names put the given name first; everything after the first token is the family
    name, so double surnames ("Davidovich Fokina") and particles stay together.
    """
    tokens = full_name.split()
    if len(tokens) <= 1:
        return full_name.strip()
    return " ".join(tokens[1:])


def display_label(full_name: str | None) -> str | None:
    """Short viewer label: the family name, abbreviated when it is longer than 8 letters.

    "Sabalenka" -> "Sab."; "de Minaur" stays (8 letters); a long hyphenated name keeps both
    parts ("Auger-Aliassime" -> "Aug-Ali."); a long double surname uses its first word when
    that fits ("Mpetshi Perricard" -> "Mpetshi"), else the first 3 letters of it ("Dav.").
    """
    if not full_name or not full_name.strip():
        return None
    family = surname(full_name)
    if _letters(family) <= LABEL_MAX_LETTERS:
        return _cap(family)
    words = family.split()
    while len(words) > 1 and words[0].lower() in PARTICLES:
        words = words[1:]
    head = words[0]
    if "-" in head and len(words) == 1:
        parts = [p for p in head.split("-") if p]
        return "-".join(_cap(p[:3]) for p in parts) + "."
    if _letters(head) <= LABEL_MAX_LETTERS:
        return _cap(head)
    return _cap(head[:3]) + "."


def display_labels(names: tuple[str, str]) -> tuple[str, str]:
    """Labels for both players, with the given-name initial added when they would collide."""
    first, second = (display_label(n) or "" for n in names)
    if first.lower() == second.lower():
        first, second = (
            f"{n.split()[0][:1].upper()}. {lab}" for n, lab in zip(names, (first, second))
        )
    return first, second


# --------------------------------------------------------------------------- metadata -> rows


def metadata_row(match_id: str, metadata: Path) -> dict | None:
    if not metadata.is_file():
        return None
    with metadata.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("match_key") == match_id:
                return row
    return None


def parse_sets(score: str) -> list[tuple[int, int]]:
    """Metadata score "6-3 6-7(5) 6-4" -> [(6, 3), (6, 7), (6, 4)] (winner first)."""
    sets = []
    for part in str(score or "").split():
        match = re.fullmatch(r"(\d+)-(\d+)(?:\(\d+\))?", part)
        if match:
            sets.append((int(match.group(1)), int(match.group(2))))
    return sets


def _scored(row: dict) -> bool:
    return str(row.get("server", "")).strip() in ("1", "2") and bool(row.get("games_1"))


def _completed(row: dict) -> list[tuple[int, int]]:
    return [
        tuple(int(v) for v in entry.split("-"))
        for entry in str(row.get("completed_sets") or "").split(";")
        if re.fullmatch(r"\d+-\d+", entry)
    ]


def row_order(ledger: list[dict], final_sets: list[tuple[int, int]]) -> dict:
    """Which board row is the metadata winner, from the board scores the ledger decoded.

    Under the order "row 1 = winner" every completed set on the board must equal the metadata
    set, and the games inside a set can never exceed the metadata's final games for it. The
    reverse order is checked the same way; violations are counted per scored attempt.
    """
    violations = {"row1_winner": 0, "row1_loser": 0}
    checked = 0
    for row in ledger:
        if not _scored(row):
            continue
        set_index = int(row.get("set_number") or 0) - 1
        if not 0 <= set_index < len(final_sets):
            continue
        checked += 1
        completed = _completed(row)
        games = (int(row["games_1"]), int(row["games_2"]))
        for order, flip in (("row1_winner", False), ("row1_loser", True)):
            target = [(b, a) if flip else (a, b) for a, b in final_sets]
            bad = any(i >= len(target) or tuple(s) != target[i] for i, s in enumerate(completed))
            bad = bad or games[0] > target[set_index][0] or games[1] > target[set_index][1]
            violations[order] += int(bad)
    best = min(violations, key=violations.get)
    other = "row1_loser" if best == "row1_winner" else "row1_winner"
    margin = violations[other] - violations[best]
    decided = (
        checked > 0
        and margin >= MIN_ROW_MARGIN
        and violations[best] <= MAX_ROW_VIOLATION_SHARE * checked
    )
    return {
        "order": best if decided else None,
        "attempts_checked": checked,
        "violations": violations,
        "confidence": round(1 - violations[best] / checked, 4) if decided else 0.0,
    }


# --------------------------------------------------------------------------- changeovers


def _tiebreak_points(row: dict) -> int:
    return sum(int(v) for v in (row.get("points_1"), row.get("points_2")) if str(v).isdigit())


def tiebreak_lengths(score: str | None) -> dict[int, int]:
    """set number -> points in its tiebreak, from a metadata score ("7-6(5)" -> 7-5 = 12)."""
    out = {}
    for number, part in enumerate(str(score or "").split(), 1):
        match = re.fullmatch(r"\d+-\d+\((\d+)\)", part)
        if match:
            loser = int(match.group(1))
            out[number] = loser + max(7, loser + 2)
    return out


def changeover_counts(
    ledger: list[dict], tiebreaks: dict[int, int] | None = None
) -> dict[int, int]:
    """attempt -> end changes before it since the start of the match (ITF rule 10).

    A finished tiebreak's length comes from the metadata score when given (``tiebreaks``);
    the board resets before its last points are read, so the longest one seen in the set is
    only a fallback (as in ``player_identity.parity_from_scores``) and can miss a change. This
    keeps absolute counts, not parity, so two attempts can be compared ("no end change
    between them") rather than only matched mod 2.
    """
    scored = [r for r in ledger if _scored(r)]
    # ``changeovers_before`` takes each finished tiebreak's TOTAL points and counts only the
    # changes strictly inside it (the change after its last point is the set-end change). The
    # board never shows a tiebreak's final score, so the longest score seen is the points
    # before the last point: the total is at least that plus one.
    longest: dict[int, int] = dict(tiebreaks or {})
    for row in scored:
        if (row["games_1"], row["games_2"]) == ("6", "6"):
            number = int(row.get("set_number") or 0)
            if number not in (tiebreaks or {}):
                longest[number] = max(longest.get(number, 0), _tiebreak_points(row) + 1)
    counts = {}
    for row in scored:
        completed = [a + b for a, b in _completed(row)]
        games = int(row["games_1"]) + int(row["games_2"])
        counts[int(row["pt"])] = changeovers_before(
            completed,
            games,
            tiebreak_points=_tiebreak_points(row) if games == TIEBREAK_GAMES - 1 else 0,
            completed_tiebreak_points=[
                longest.get(index + 1, ASSUMED_TIEBREAK_POINTS)
                for index in range(len(completed))
            ],
        )
    return counts


def placements(ledger: list[dict], tiebreaks: dict[int, int] | None = None) -> dict[int, dict]:
    """attempt -> how the changeover rules place it: end-change count, server row, source.

    An abstained attempt takes its point's score (same ends, same server); else the count of
    its scored neighbours when both have the same count (no end change can fall between them),
    or the count 0 when it precedes a first scored attempt that is still before any change.
    """
    rows = sorted(ledger, key=lambda r: int(r["pt"]))
    counts = changeover_counts(rows, tiebreaks)
    out: dict[int, dict] = {}
    by_point: dict[str, dict] = {}
    for row in rows:
        if _scored(row):
            pt = int(row["pt"])
            out[pt] = {
                "changeovers": counts[pt],
                "server_row": int(row["server"]),
                "placed_by": "own_score",
                "set_number": row.get("set_number"),
                "score_row": pt,
                "score_confidence": _float(row.get("score_confidence")),
            }
            if row.get("point_index"):
                by_point.setdefault(row["point_index"], out[pt])
    for index, row in enumerate(rows):
        pt = int(row["pt"])
        if pt in out:
            continue
        sibling = by_point.get(row.get("point_index") or "")
        if sibling is not None:
            out[pt] = {**sibling, "placed_by": "same_point"}
            continue
        before = next((r for r in reversed(rows[:index]) if _scored(r)), None)
        after = next((r for r in rows[index + 1 :] if _scored(r)), None)
        if after is None:
            continue
        a = out[int(after["pt"])]
        if before is not None:
            b = out[int(before["pt"])]
            if b["changeovers"] != a["changeovers"]:
                continue
            same_game = (
                before["set_number"] == after["set_number"]
                and (before["games_1"], before["games_2"]) == (after["games_1"], after["games_2"])
                and (before["games_1"], before["games_2"]) != ("6", "6")
            )
            out[pt] = {
                "changeovers": a["changeovers"],
                "server_row": a["server_row"] if same_game else None,
                "placed_by": "scored_neighbours",
                "set_number": after.get("set_number"),
                "score_row": [int(before["pt"]), int(after["pt"])],
                "score_confidence": min(b["score_confidence"], a["score_confidence"]),
            }
        elif a["changeovers"] == 0:
            out[pt] = {
                "changeovers": 0,
                "server_row": None,
                "placed_by": "before_first_change",
                "set_number": after.get("set_number"),
                "score_row": int(after["pt"]),
                "score_confidence": a["score_confidence"],
            }
    return out


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------- evidence


def attempt_number(key: str) -> int | None:
    match = re.search(r"__pt(\d+)", key or "")
    return int(match.group(1)) if match else None


def side_of_y(y: float | None) -> str | None:
    if y is None:
        return None
    return "near" if float(y) < NET_Y_M else "far"


def serve_sides_from_result(result: dict) -> dict[int, str]:
    """attempt -> side of the accepted serve flight's start (``MATCH_RESULT`` attempts)."""
    sides: dict[int, str] = {}
    for attempt in result.get("attempts") or []:
        pt = attempt_number(attempt.get("key", ""))
        for flight in attempt.get("accepted_flights") or []:
            if flight.get("role") == "serve" and flight.get("start_xyz_m") and pt is not None:
                sides.setdefault(pt, side_of_y(flight["start_xyz_m"][1]))
                break
    return sides


def serve_sides_from_documents(documents: list[Path]) -> dict[int, str]:
    """attempt -> side of the fitted serve contact in the exported portal document."""
    sides: dict[int, str] = {}
    for path in documents:
        if "__component_" in path.name:
            continue
        pt = attempt_number(path.name)
        for contact in json.loads(path.read_text()).get("contacts") or []:
            if contact.get("phase") == "serve" and contact.get("side") in ("near", "far"):
                if pt is not None:
                    sides.setdefault(pt, contact["side"])
                break
    return sides


def merge_serve_sides(*sources: dict[int, str]) -> tuple[dict[int, str], list[int]]:
    """One side per attempt; attempts whose sources disagree are dropped and listed."""
    merged: dict[int, str] = {}
    conflicts: set[int] = set()
    for source in sources:
        for pt, side in source.items():
            if side not in ("near", "far"):
                continue
            if pt in merged and merged[pt] != side:
                conflicts.add(pt)
            merged.setdefault(pt, side)
    return {k: v for k, v in merged.items() if k not in conflicts}, sorted(conflicts)


# --------------------------------------------------------------------------- resolution


def resolve(
    match_id: str,
    ledger: list[dict],
    serve_sides: dict[int, str],
    metadata: dict | None,
) -> dict:
    """Match-level assignment plus per-attempt identity with evidence and source."""
    report: dict = {"schema": SCHEMA, "match_id": match_id}
    roster = roster_from_match_id(match_id)
    names = (metadata["winner_name"], metadata["loser_name"]) if metadata else None
    report["metadata"] = (
        {"winner": names[0], "loser": names[1], "score": metadata.get("score")} if names else None
    )
    report["roster_from_match_id"] = list(roster.names) if roster.names else None
    order = row_order(ledger, parse_sets(metadata["score"])) if metadata else {"order": None}
    report["row_anchor"] = order
    row_names: dict[int, str] = {}
    if names and order.get("order"):
        winner_row = 1 if order["order"] == "row1_winner" else 2
        row_names = {winner_row: names[0], 3 - winner_row: names[1]}
    labels: dict[int, str] = {}
    if row_names:
        pair = display_labels((row_names[1], row_names[2]))
        labels = {1: pair[0], 2: pair[1]}
    report["row_names"] = {str(k): v for k, v in row_names.items()} or None
    report["row_labels"] = {str(k): v for k, v in labels.items()} or None

    placed = placements(ledger, tiebreak_lengths(metadata.get("score")) if metadata else None)
    # One bit for the match: did row 1 start the match at the near end?
    votes = Counter()
    set_votes: dict[str, Counter] = {}
    near_seen = 0
    for pt, side in serve_sides.items():
        place = placed.get(pt)
        if place is None or place["placed_by"] != "own_score":
            continue
        row1_near_now = (place["server_row"] == 1) == (side == "near")
        bit = bool(row1_near_now) ^ bool(place["changeovers"] % 2)
        votes[bit] += 1
        set_votes.setdefault(str(place.get("set_number")), Counter())[bit] += 1
        near_seen += side == "near"
    total = votes[True] + votes[False]
    start = votes[True] >= votes[False]
    agreement = round(max(votes[True], votes[False]) / total, 4) if total else 0.0
    side_decided = total >= MIN_SIDE_VOTES and agreement >= MIN_SIDE_AGREEMENT
    sets = {}
    conflicting_sets = set()
    for number, counter in sorted(set_votes.items()):
        count = counter[True] + counter[False]
        share = round(counter[start] / count, 4) if count else None
        conflict = count >= MIN_SET_VOTES and share < MIN_SET_AGREEMENT
        sets[number] = {"votes": count, "agreement_with_match": share, "conflict": conflict}
        if conflict:
            conflicting_sets.add(number)
    report["side_anchor"] = {
        "row1_near_at_match_start": start if side_decided else None,
        "serve_votes": total,
        "votes_for": max(votes[True], votes[False]),
        "votes_against": min(votes[True], votes[False]),
        "agreement": agreement,
        # Diagnostic only: far from 0.5 when the serve detector or the fits favour one end
        # (selection), which the vote tolerates; stuck side evidence shows as low agreement.
        "observed_near_share": round(near_seen / total, 4) if total else None,
        "sets": sets,
    }
    failures = []
    if not names:
        failures.append("match not in metadata")
    elif not order.get("order"):
        failures.append("board scores do not settle which row is the metadata winner")
    if not side_decided:
        failures.append(
            f"serve sides do not settle the ends ({total} votes, agreement {agreement})"
        )
    report["status"] = "named" if not failures else "unresolved"
    report["failures"] = failures
    match_confidence = round(order.get("confidence", 0.0) * agreement, 4) if not failures else 0.0
    report["confidence"] = match_confidence

    attempts = {}
    for row in sorted(ledger, key=lambda r: int(r["pt"])):
        pt = int(row["pt"])
        observed = serve_sides.get(pt)
        place = placed.get(pt)
        entry: dict = {
            "source": "unresolved",
            "near": None,
            "far": None,
            "near_label": None,
            "far_label": None,
            "server": None,
            "server_side": None,
            "confidence": 0.0,
            "serve_side_observed": observed,
            "placed_by": place["placed_by"] if place else None,
            "disagreement": False,
        }
        if place and row_names and place.get("server_row"):
            entry["server"] = row_names[place["server_row"]]
        if failures:
            entry["reason"] = "; ".join(failures)
        elif place is None:
            entry["reason"] = "no decoded score places this attempt between end changes"
        elif str(place.get("set_number")) in conflicting_sets:
            entry["set_conflict"] = True
            entry["reason"] = "this set's serve sides contradict the match end-change assignment"
        else:
            row1_near = start ^ bool(place["changeovers"] % 2)
            near_row = 1 if row1_near else 2
            expected = None
            if place.get("server_row"):
                expected = "near" if place["server_row"] == near_row else "far"
            entry.update(
                changeovers=place["changeovers"],
                score_row=place["score_row"],
                server_row=place.get("server_row"),
                serve_side_expected=expected,
            )
            disagree = observed is not None and expected is not None and observed != expected
            direct = observed is not None and place["placed_by"] == "own_score" and not disagree
            factor = 1.0 if direct else max(place["score_confidence"], 0.5)
            if place["placed_by"] in ("scored_neighbours", "before_first_change"):
                factor *= 0.9
            if disagree:
                # The rules win an isolated disagreement: in the eye audit the fitted serve
                # side was the wrong witness on nearly every one (a rally shot taken for the
                # serve). A set where serves disagree wholesale is withheld above instead.
                factor *= 0.5
                entry["disagreement"] = True
                entry["reason"] = "observed serve side contradicts the rules; rules kept"
            entry.update(
                source="direct" if direct else "propagated",
                near=row_names[near_row],
                far=row_names[3 - near_row],
                near_label=labels[near_row],
                far_label=labels[3 - near_row],
                server_side=expected,
                confidence=round(match_confidence * factor, 4),
            )
        attempts[str(pt)] = entry
    report["attempts"] = attempts
    report["counts"] = summary_counts(attempts)
    return report


def summary_counts(attempts: dict) -> dict:
    counts = Counter(e["source"] for e in attempts.values())
    total = len(attempts)
    return {
        "attempts": total,
        **{source: counts.get(source, 0) for source in SOURCES},
        **{
            f"{source}_share": round(counts.get(source, 0) / total, 4) if total else None
            for source in SOURCES
        },
        "evidence_rule_disagreements": sum(e["disagreement"] for e in attempts.values()),
        "set_conflict_attempts": sum(bool(e.get("set_conflict")) for e in attempts.values()),
        "server_named": sum(bool(e["server"]) for e in attempts.values()),
    }


def summary(report: dict) -> dict:
    """The compact match view OUTCOMES and the receipt carry."""
    return {
        "file": FILE,
        "schema": SCHEMA,
        "status": report["status"],
        "failures": report["failures"],
        "confidence": report["confidence"],
        "row_names": report["row_names"],
        "row_anchor": {k: report["row_anchor"].get(k) for k in ("order", "violations")},
        "side_anchor": report["side_anchor"],
        "serve_side_conflicts": len(report.get("serve_side_conflicts") or []),
        "counts": report["counts"],
    }


# --------------------------------------------------------------------------- outputs


def identity_record(entry: dict | None) -> dict:
    """Per-attempt identity for data rows (MATCH_RESULT, OUTCOMES)."""
    if not entry:
        return {"source": "unresolved", "near": None, "far": None, "server": None}
    keep = (
        "source",
        "near",
        "far",
        "server",
        "server_side",
        "confidence",
        "placed_by",
        "changeovers",
        "serve_side_observed",
        "serve_side_expected",
        "disagreement",
        "set_conflict",
        "reason",
    )
    return {k: entry[k] for k in keep if k in entry}


def players_block(entry: dict | None) -> dict:
    """The ``players`` fields a portal document carries: names (or nulls), labels, evidence."""
    named = bool(entry and entry.get("near"))
    return {
        "names": {"near": entry["near"] if named else None, "far": entry["far"] if named else None},
        "labels": {
            "near": entry["near_label"] if named else None,
            "far": entry["far_label"] if named else None,
        },
        "name_evidence": {"schema": SCHEMA, **(entry or {"source": "unresolved"})},
    }


def _player_on(entry: dict | None, side: str | None) -> str | None:
    if not entry or side not in ("near", "far"):
        return None
    return entry.get(side)


def stamp_documents(documents: list[Path], report: dict) -> int:
    for path in documents:
        document = json.loads(path.read_text())
        entry = report["attempts"].get(str(attempt_number(path.name)))
        document.setdefault("players", {}).update(players_block(entry))
        for contact in document.get("contacts") or []:
            contact["player"] = _player_on(entry, contact.get("side"))
        path.write_text(json.dumps(document, separators=(",", ":"), allow_nan=False))
    return len(documents)


def stamp_result(result: dict, report: dict) -> dict:
    """``MATCH_RESULT`` attempts get ``player_identity``; accepted flights get their hitter."""
    result["player_identity"] = summary(report)
    for attempt in result.get("attempts") or []:
        entry = report["attempts"].get(str(attempt_number(attempt.get("key", ""))))
        attempt["player_identity"] = identity_record(entry)
        for flight in attempt.get("accepted_flights") or []:
            side = side_of_y((flight.get("start_xyz_m") or [None, None])[1])
            flight["hitter_side"] = side
            flight["hitter"] = _player_on(entry, side)
    return result


def match_documents(portal: Path | None, match_id: str) -> list[Path]:
    if portal is None or not (portal / "data").is_dir():
        return []
    return sorted(
        p for p in (portal / "data").glob("*.json") if match_id in p.name and attempt_number(p.name)
    )


def read_ledger(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def decide(
    match_id: str, ledger_path: Path, metadata_path: Path, result: dict, documents: list[Path]
) -> dict:
    sides, conflicts = merge_serve_sides(
        serve_sides_from_documents(documents), serve_sides_from_result(result)
    )
    report = resolve(match_id, read_ledger(ledger_path), sides, metadata_row(match_id, metadata_path))
    report["serve_side_conflicts"] = conflicts
    report["inputs"] = {
        "ledger": str(ledger_path),
        "metadata": str(metadata_path),
        "serve_sides": "portal document serve contacts + MATCH_RESULT accepted serve flights",
        "portal_documents": len(documents),
    }
    report["built"] = time.time()
    return report


def run(out: Path, match_id: str, metadata_path: Path, portal: Path | None = None) -> dict:
    """Decide, then write PLAYER_IDENTITY.json, MATCH_RESULT, portal documents and summary."""
    result_path = out / "MATCH_RESULT.json"
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    documents = match_documents(portal, match_id)
    report = decide(match_id, out / "upstream" / match_id / LEDGER, metadata_path, result, documents)
    (out / FILE).write_text(json.dumps(report, indent=1) + "\n")
    if result:
        result_path.write_text(json.dumps(stamp_result(result, report), indent=1) + "\n")
    stamped = stamp_documents(documents, report)
    if portal is not None and (portal / "data").is_dir():
        (portal / "data" / "player_names.json").write_text(json.dumps(report, indent=1) + "\n")
    return {**summary(report), "documents_stamped": stamped}


# --------------------------------------------------------------------------- pooled report


def pool(run_dirs: list[Path], metadata_path: Path) -> dict:
    """Read-only identity over published matches (``ARCHIVE_RUN/MATCH`` trees)."""
    matches = []
    for run_dir in run_dirs:
        for match in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            ledger = match / "upstream" / match.name / LEDGER
            result_path = match / "MATCH_RESULT.json"
            if not ledger.is_file() or not result_path.is_file():
                continue
            documents = match_documents(match / "portal" / "3d", match.name)
            report = decide(
                match.name, ledger, metadata_path, json.loads(result_path.read_text()), documents
            )
            matches.append({"run": run_dir.name, "match_id": match.name, **summary(report)})
    pooled = Counter()
    for match in matches:
        for key in (
            "attempts",
            *SOURCES,
            "evidence_rule_disagreements",
            "set_conflict_attempts",
            "server_named",
        ):
            pooled[key] += match["counts"][key]
    total = pooled["attempts"]
    return {
        "schema": f"{SCHEMA}_pool",
        "runs": [str(r) for r in run_dirs],
        "matches": matches,
        "matches_named": sum(m["status"] == "named" for m in matches),
        "pooled": {
            **dict(pooled),
            **{f"{s}_share": round(pooled[s] / total, 4) if total else None for s in SOURCES},
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, help="match output directory (writes outputs)")
    parser.add_argument("--match-id")
    parser.add_argument(
        "--portal", type=Path, help="staged portal to stamp (default OUT/portal/3d)"
    )
    parser.add_argument("--pool", type=Path, nargs="+", help="published run dirs (read-only)")
    parser.add_argument("--report", type=Path, help="pooled report path")
    parser.add_argument("--metadata", type=Path, help="matches.csv (default: shared metadata)")
    args = parser.parse_args(argv)
    from scripts.shared_data import repository_root, resolve_shared_root

    metadata = args.metadata or resolve_shared_root(repository_root(), None) / METADATA
    if args.pool:
        report = pool(args.pool, metadata)
        text = json.dumps(report, indent=1) + "\n"
        if args.report:
            args.report.write_text(text)
        print(json.dumps({"matches": len(report["matches"]), **report["pooled"]}, indent=1))
        return 0
    if not args.out or not args.match_id:
        parser.error("--out and --match-id are required unless --pool is given")
    portal = args.portal or args.out / "portal" / "3d"
    print(json.dumps(run(args.out, args.match_id, metadata, portal), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
