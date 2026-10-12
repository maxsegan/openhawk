"""Missed active play: stretches of a broadcast where play likely happened but no attempt exists.

    python -m cv.pipeline.missed_play UPSTREAM_MATCH_DIR [--out MISSED_PLAY.json]

Owner request 2026-10-05. Whole matches have lost most of their points without any error: an
audio hole stopped serve audio 30 min into Tiafoe (dry run 3), a single-camera feed was called
a graphic (Mladenovic-Zhang), and Etcheverry-Atmane finds a serve for only 60% of its score
changes (``cv/experiments/point_finding_gap``, ``long_shot_view``). ``MISSED_PLAY.json``
(``missed_play_v1``) lists, per match, the stretches the point ledger has no attempt for while
other whole-match signals say a point was played:

- **score changes** with no attempt ending in the ``SCORE_LAG_SECONDS`` before them (the
  board moves after the point; a change inside an attempt is explained by it);
- **play-view time** in a gap between attempts far longer than the match's typical attempt
  spacing (``shot_views_v1.csv``, split windows of ``view_windows_v1.csv`` where present);
- **hit rhythm**: chains of serve-audio peaks at rally spacing (``serve_audio_peaks_v1.csv``);
  weak alone (on clean matches about 4x as frequent inside attempts as in non-play time, and
  rare), so it only supports other evidence;
- **serve candidates** the detector wrote (``serve_attempts_v1.csv``) that the ledger dropped;
- **coverage**: whole-source readers that stopped short, from their coverage records
  (``source_coverage``) and, for matches older than the guard, from the last timestamp each
  stream reached against the source duration.

Each flag has a time range, the evidence, a confidence (``high``/``medium``/``low``) and a
likely cause. A flag is a pointer for review, not a count of missed points. Thresholds were set
on development matches (dry runs 3-5; ``cv/experiments/failure_index/REPORT.md``). Reads only
upstream point-finding outputs; no evaluation label is read and nothing here changes a flight.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

SCHEMA = "missed_play_v1"
FILE = "MISSED_PLAY.json"
LEDGER_NAME = "automatic_point_ledger_v1.csv"
#: A point's attempt ends at most this long before the scoreboard changes (long_shot_view used
#: 75 s for its point-recall proxy; replays and slow boards stretch it).
SCORE_LAG_SECONDS = 75.0
#: A gap between attempts is long when it exceeds this many typical spacings and this floor.
GAP_FACTOR = 4.0
GAP_FLOOR_SECONDS = 150.0
#: Play-view seconds inside a long gap that count as evidence of play.
PLAY_MIN_SECONDS = 45.0
#: Margins around attempts not counted as gap play (walk-back, serve setup).
GAP_MARGIN_SECONDS = 10.0
#: Hit rhythm: at least this many peaks at rally spacing with a regular period.
RHYTHM_MIN_PEAKS = 5
RHYTHM_INTERVAL = (0.7, 2.0)
RHYTHM_MAX_CV = 0.35
#: Coverage: a stream ending earlier than this before the source end is short
#: (``source_coverage``'s rule, max(30 s, 1 %)).
COVERAGE_MIN_SECONDS = 30.0
COVERAGE_SHARE = 0.01
#: A camera shot this long marks a continuous feed (``view_classifier`` CONTINUOUS_FEED_SECONDS).
CONTINUOUS_FEED_SECONDS = 1200.0
#: ``read_age_seconds`` of a decoded state no scoreboard read supports yet.
UNREAD_AGE_SECONDS = 1e5
#: Before the first and after the last attempt (walk-on, warm-up, ceremonies, final-score
#: graphics) a flag needs this many unexplained score changes and this much play to rise above
#: ``low``; coverage evidence is unaffected. Two pre-match flags were false in the eye check.
EDGE_MIN_CHANGES = 3
EDGE_MIN_PLAY_SECONDS = 60.0
#: Matches shorter than this with no attempt are not flagged as a whole (clips, highlights).
EMPTY_MATCH_MIN_SECONDS = 600.0
CONFIDENCES = ("high", "medium", "low")
#: Small JSON files that may hold a ``source_coverage`` record.
COVERAGE_FILES = (
    "shot_boundaries_v1.json",
    "score_crops_native.coordinates.json",
    "overlay_crops.coordinates.json",
    "frames_native.json",
)
#: Upstream files this module reads (backfill extracts exactly these).
INPUT_NAMES = (
    LEDGER_NAME,
    "score_runs.csv",
    "score_state_v1.csv",
    "score_state_v1.json",
    "shot_views_v1.csv",
    "view_windows_v1.csv",
    "shot_boundaries_v1.csv",
    "serve_audio_peaks_v1.csv",
    "serve_attempts_v1.csv",
    "source_integrity.json",
    *COVERAGE_FILES,
)


# --------------------------------------------------------------------------- inputs


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def attempts(upstream: Path) -> np.ndarray:
    """Ledger attempt windows ``[[t0, t1], ...]`` (source seconds), sorted by start."""
    spans = [
        (_float(r.get("rally_t_start")), _float(r.get("rally_t_end")))
        for r in _rows(upstream / LEDGER_NAME)
    ]
    spans = sorted((a, b) for a, b in spans if a is not None and b is not None)
    return np.array(spans, dtype=float).reshape(-1, 2)


def score_changes(upstream: Path) -> list[tuple[float, float]]:
    """``(t, read_age)`` of every point transition of the grammar-decoded score.

    ``score_state_v1.csv`` (one decoded state a second, ``score_grammar_decode_v2``) moves only
    on transitions the score grammar accepts; the raw ``score_runs.csv`` also changes when the
    board switches between point and set-summary displays, so it is used only without it.
    ``read_age`` is how old the read behind the new state was (the change may be that late).
    """
    keys = ("sets_won_1", "sets_won_2", "games_1", "games_2", "points_1", "points_2")
    changes, previous = [], None
    states = _rows(upstream / "score_state_v1.csv")
    if not states:
        keys = ("g1", "g2", "p1", "p2", "sets1", "sets2")
        states = [{**r, "t": r.get("t_start")} for r in _rows(upstream / "score_runs.csv")]
    unread = True
    for row in states:
        state = tuple(row.get(k) for k in keys)
        t = _float(row.get("t"))
        if t is None:
            continue
        age = _float(row.get("read_age_seconds")) or 0.0
        # The decoder's first state before any read is a prior; its first real read is not
        # a transition.
        if previous is not None and state != previous and not unread:
            changes.append((t, min(age, SCORE_LAG_SECONDS)))
        unread = unread and age >= UNREAD_AGE_SECONDS
        previous = state
    return changes


def scoreboard_read_share(upstream: Path) -> float | None:
    path = upstream / "score_state_v1.json"
    if not path.is_file():
        return None
    document = _load(path)
    observations = document.get("observations") or 0
    return round((document.get("usable_reads") or 0) / observations, 3) if observations else None


def view_units(upstream: Path) -> list[tuple[float, float, str]]:
    """``(t0, t1, view)`` per camera shot; a split long shot contributes its windows instead."""
    shots = _rows(upstream / "shot_views_v1.csv")
    windows = _rows(upstream / "view_windows_v1.csv")
    split = {w.get("shot_index") for w in windows}
    units = [
        (float(s["t_start"]), float(s["t_end"]), s.get("view") or "")
        for s in shots
        if s.get("shot_index") not in split
    ]
    units += [(float(w["t_start"]), float(w["t_end"]), w.get("view") or "") for w in windows]
    return sorted(units)


def longest_shot_seconds(upstream: Path) -> float | None:
    rows = _rows(upstream / "shot_views_v1.csv") or _rows(upstream / "shot_boundaries_v1.csv")
    lengths = [
        float(r["t_end"]) - float(r["t_start"]) for r in rows if "t_start" in r and "t_end" in r
    ]
    return round(max(lengths), 1) if lengths else None


def audio_peaks(upstream: Path) -> np.ndarray:
    rows = _rows(upstream / "serve_audio_peaks_v1.csv")
    return np.array([(float(r["t"]), float(r["z"])) for r in rows], dtype=float).reshape(-1, 2)


def serve_candidates(upstream: Path) -> list[float]:
    return [
        t
        for r in _rows(upstream / "serve_attempts_v1.csv")
        if (t := _float(r.get("rally_t_start"))) is not None
    ]


def source_duration(upstream: Path) -> float | None:
    path = upstream / "source_integrity.json"
    if not path.is_file():
        return None
    return _float(_load(path).get("duration_seconds"))


# --------------------------------------------------------------------------- evidence


def _inside(times: np.ndarray, spans: np.ndarray) -> np.ndarray:
    if not len(times) or not len(spans):
        return np.zeros(len(times), dtype=bool)
    starts = np.searchsorted(spans[:, 0], times, side="right") - 1
    ok = starts >= 0
    result = np.zeros(len(times), dtype=bool)
    result[ok] = times[ok] <= spans[starts[ok], 1]
    return result


def rhythm_chains(peaks: np.ndarray, attempt_spans: np.ndarray) -> list[tuple[float, float]]:
    """Runs of serve-audio peaks at rally spacing with a regular period (hit rhythm).

    Peaks below the median strength of the peaks inside attempts are ignored (all peaks when
    there is no attempt), so the threshold follows the broadcast's own audio scale.
    """
    if len(peaks) < RHYTHM_MIN_PEAKS:
        return []
    inside = _inside(peaks[:, 0], attempt_spans)
    reference = peaks[inside, 1] if inside.sum() >= 50 else peaks[:, 1]
    strong = peaks[peaks[:, 1] >= float(np.median(reference))]
    chains, i = [], 0
    low, high = RHYTHM_INTERVAL
    while i < len(strong) - 1:
        j = i
        while j + 1 < len(strong) and low <= strong[j + 1, 0] - strong[j, 0] <= high:
            j += 1
        if j - i + 1 >= RHYTHM_MIN_PEAKS:
            gaps = np.diff(strong[i : j + 1, 0])
            if gaps.std() / gaps.mean() <= RHYTHM_MAX_CV:
                chains.append((float(strong[i, 0]), float(strong[j, 0])))
        i = j + 1
    return chains


def coverage_records(upstream: Path) -> list[dict]:
    """Every ``source_coverage`` record the point-finding readers left (receipts, sidecars)."""
    found = []

    def walk(node, where: str) -> None:
        if isinstance(node, dict):
            if "reached_seconds" in node and "expected_seconds" in node:
                found.append({**node, "where": where})
                return
            for value in node.values():
                walk(value, where)
        elif isinstance(node, list):
            for value in node:
                walk(value, where)

    paths = [upstream / name for name in COVERAGE_FILES]
    paths += sorted((upstream / "run_manifests").glob("*.json"))
    for path in paths:
        if path.is_file() and path.stat().st_size < 8 << 20:
            try:
                walk(_load(path), path.name)
            except (OSError, ValueError):
                continue
    return found


def _short(reached: float, expected: float) -> bool:
    return expected - reached > max(COVERAGE_MIN_SECONDS, COVERAGE_SHARE * expected)


def coverage_gaps(upstream: Path, duration: float | None) -> list[dict]:
    """Streams that stopped short: recorded coverage records, then derived last timestamps."""
    gaps = []
    for record in coverage_records(upstream):
        reached, expected = float(record["reached_seconds"]), float(record["expected_seconds"])
        if record.get("complete") is False or _short(reached, expected):
            gaps.append(
                {
                    "reader": record.get("reader") or record["where"],
                    "reached_seconds": round(reached, 1),
                    "expected_seconds": round(expected, 1),
                    "origin": f"coverage record ({record['where']})",
                }
            )
    if not duration:
        return gaps
    recorded = {g["reader"] for g in gaps}
    # Older matches have no coverage records: the last time each stream reached stands in.
    peaks = audio_peaks(upstream)
    shots = _rows(upstream / "shot_views_v1.csv")
    derived = {
        "serve_audio": float(peaks[-1, 0]) if len(peaks) else None,
        "shot_signals": max((float(r["t_end"]) for r in shots), default=None),
    }
    for reader, reached in derived.items():
        if reached is None or reader in recorded or not _short(reached, duration):
            continue
        gaps.append(
            {
                "reader": reader,
                "reached_seconds": round(reached, 1),
                "expected_seconds": round(duration, 1),
                "origin": "derived: last timestamp of the stream",
            }
        )
    return gaps


def _seconds_of(view: str, units: list[tuple[float, float, str]], lo: float, hi: float) -> float:
    return sum(max(0.0, min(hi, b) - max(lo, a)) for a, b, v in units if v == view and b > lo)


def merge(spans: np.ndarray) -> np.ndarray:
    """Overlapping attempt windows joined (sorted input)."""
    merged: list[list[float]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([float(a), float(b)])
    return np.array(merged, dtype=float).reshape(-1, 2)


def typical_spacing(spans: np.ndarray) -> float | None:
    """Median start-to-start spacing of attempts within play (gaps over 180 s left out)."""
    if len(spans) < 5:
        return None
    gaps = np.diff(spans[:, 0])
    gaps = gaps[gaps <= 180.0]
    return round(float(np.median(gaps)), 1) if len(gaps) else None


# --------------------------------------------------------------------------- flags


def _cause(evidence: dict, context: dict) -> str:
    if evidence.get("coverage"):
        return "source_read_short:" + ",".join(sorted({g["reader"] for g in evidence["coverage"]}))
    if context["attempts"] == 0 and (context["longest_shot_seconds"] or 0) >= (
        CONTINUOUS_FEED_SECONDS
    ):
        return "continuous_feed_view_gating"
    if evidence.get("serve_candidates_dropped"):
        return "ledger_dropped_serve_candidates"
    span = max(1.0, evidence["t1"] - evidence["t0"])
    if evidence.get("score_changes") and evidence["play_view_seconds"] < 0.1 * span:
        return "not_play_view"
    return "serve_detection_miss"


def _confidence(evidence: dict, context: dict) -> str:
    changes = len(evidence.get("score_changes") or [])
    play = evidence["play_view_seconds"]
    rhythm = evidence["hit_rhythm_chains"]
    if evidence.get("coverage"):
        return "high"
    if evidence.get("edge") and not evidence.get("whole_match"):
        if changes >= EDGE_MIN_CHANGES and play >= EDGE_MIN_PLAY_SECONDS:
            return "medium"
        return "low"
    if context["attempts"] == 0 and evidence.get("whole_match"):
        return "high" if (changes or play >= PLAY_MIN_SECONDS or rhythm >= 3) else "medium"
    if changes >= 2 and (play >= 30.0 or rhythm >= 1):
        return "high"
    if changes >= 2 or (changes == 1 and (play >= 20.0 or rhythm >= 1)):
        return "medium"
    if (
        evidence.get("long_gap")
        and play >= PLAY_MIN_SECONDS
        and (rhythm >= 1 or evidence.get("serve_candidates_dropped"))
    ):
        return "medium"
    return "low"


def build(upstream: Path, match_id: str | None = None) -> dict:
    """The missed-play record of one upstream match folder (``upstream/<match_id>``)."""
    upstream = Path(upstream)
    spans = attempts(upstream)
    duration = source_duration(upstream)
    changes = score_changes(upstream)
    units = view_units(upstream)
    peaks = audio_peaks(upstream)
    merged = merge(spans)
    chains = rhythm_chains(peaks, merged)
    candidates = serve_candidates(upstream)
    coverage = coverage_gaps(upstream, duration)
    end = duration or max(
        [float(spans[-1, 1]) if len(spans) else 0.0]
        + [float(peaks[-1, 0]) if len(peaks) else 0.0]
        + [b for _, b, _ in units]
    )
    spacing = typical_spacing(spans)
    long_gap = max(GAP_FLOOR_SECONDS, GAP_FACTOR * spacing) if spacing else GAP_FLOOR_SECONDS
    context = {
        "attempts": int(len(spans)),
        "longest_shot_seconds": longest_shot_seconds(upstream),
    }
    change_times = np.array([t for t, _ in changes], dtype=float)
    # A change is explained by an attempt that contains it or ended shortly before it (the
    # window grows by the age of the read behind it).
    explained = np.zeros(len(change_times), dtype=bool)
    if len(spans):
        for index, (t, age) in enumerate(changes):
            explained[index] = bool(
                ((spans[:, 0] <= t) & (spans[:, 1] >= t - SCORE_LAG_SECONDS - age)).any()
            )
    unexplained = change_times[~explained]
    # Gaps: before the first attempt, between attempts, after the last.
    edges = [0.0] + [x for span in merged for x in span] + [end]
    gaps = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2) if edges[i + 1] > edges[i]]
    flags = []
    for index, (lo, hi) in enumerate(gaps):
        head, tail = index == 0 and lo == 0.0, hi == end
        inner_lo = lo if head else lo + GAP_MARGIN_SECONDS
        inner_hi = hi if tail else hi - GAP_MARGIN_SECONDS
        if inner_hi <= inner_lo:
            continue
        in_gap = unexplained[(unexplained > lo) & (unexplained <= hi)]
        cover = [g for g in coverage if g["reached_seconds"] < hi and g["expected_seconds"] > lo]
        play = _seconds_of("play", units, inner_lo, inner_hi)
        gap_chains = [c for c in chains if inner_lo <= c[0] <= inner_hi]
        dropped = [t for t in candidates if inner_lo <= t <= inner_hi]
        is_long = (hi - lo) >= long_gap and not (head or tail)
        whole = len(spans) == 0 and end >= EMPTY_MATCH_MIN_SECONDS
        if not (len(in_gap) or cover or whole):
            if not (is_long and play >= PLAY_MIN_SECONDS and (gap_chains or dropped)):
                continue
        if cover or whole:
            t0 = max(lo, min([g["reached_seconds"] for g in cover] or [lo]))
            t1 = hi
        elif len(in_gap):
            t0 = max(lo, float(in_gap.min()) - SCORE_LAG_SECONDS)
            t1 = float(in_gap.max())
            if is_long and play >= PLAY_MIN_SECONDS:
                t0, t1 = inner_lo, inner_hi
        else:
            t0, t1 = inner_lo, inner_hi
        evidence = {
            "t0": round(t0, 1),
            "t1": round(t1, 1),
            "gap": [round(lo, 1), round(hi, 1)],
            "long_gap": bool(is_long),
            "score_changes": [round(float(t), 1) for t in in_gap],
            "play_view_seconds": round(_seconds_of("play", units, t0, t1), 1),
            "hit_rhythm_chains": len([c for c in chains if t0 <= c[0] <= t1]),
            "serve_candidates_dropped": [round(t, 1) for t in dropped if t0 <= t <= t1],
            "coverage": cover,
            "whole_match": bool(whole),
            "edge": "before_first_attempt" if head else ("after_last_attempt" if tail else None),
        }
        evidence["confidence"] = _confidence(evidence, context)
        evidence["likely_cause"] = _cause(evidence, context)
        flags.append(evidence)
    flagged = sum(f["t1"] - f["t0"] for f in flags)
    by_confidence = {c: sum(f["confidence"] == c for f in flags) for c in CONFIDENCES}
    causes: dict[str, int] = {}
    for flag in flags:
        causes[flag["likely_cause"]] = causes.get(flag["likely_cause"], 0) + 1
    play_total = _seconds_of("play", units, 0.0, end)
    counts = {
        "flags": len(flags),
        "by_confidence": by_confidence,
        "by_cause": dict(sorted(causes.items(), key=lambda item: -item[1])),
        "flagged_seconds": round(flagged, 1),
        "flagged_seconds_high_or_medium": round(
            sum(f["t1"] - f["t0"] for f in flags if f["confidence"] != "low"), 1
        ),
        "unexplained_score_changes": int(len(unexplained)),
        "score_changes": len(changes),
        "attempts": int(len(spans)),
        "attempts_per_score_change": round(len(spans) / len(changes), 2) if changes else None,
        "coverage_short_streams": sorted({g["reader"] for g in coverage}),
    }
    return {
        "schema": SCHEMA,
        "match_id": match_id or upstream.name,
        "built": time.time(),
        "source_seconds": round(end, 1),
        "counts": counts,
        "context": {
            **context,
            "typical_attempt_spacing_seconds": spacing,
            "long_gap_seconds": round(long_gap, 1),
            "play_view_seconds": round(play_total, 1),
            "scoreboard_read_share": scoreboard_read_share(upstream),
            "serve_audio_peaks": int(len(peaks)),
            "hit_rhythm_chains": len(chains),
            "serve_candidates": len(candidates),
            "inputs_present": sorted(n for n in INPUT_NAMES if (upstream / n).is_file()),
        },
        "rules": {
            "score_lag_seconds": SCORE_LAG_SECONDS,
            "long_gap": f"max({GAP_FLOOR_SECONDS:.0f} s, {GAP_FACTOR:g} x typical spacing)",
            "play_min_seconds": PLAY_MIN_SECONDS,
            "rhythm": f">= {RHYTHM_MIN_PEAKS} peaks {RHYTHM_INTERVAL} s apart, CV <= "
            f"{RHYTHM_MAX_CV}",
            "coverage": f"short by more than max({COVERAGE_MIN_SECONDS:.0f} s, "
            f"{COVERAGE_SHARE:.0%})",
        },
        "note": "Flags point at stretches to review; they are not counts of missed points. "
        "No evaluation label is read.",
        "evaluation_labels_read": False,
        "flags": flags,
    }


def summary(document: dict) -> dict:
    """The counts ``OUTCOMES.json`` and the receipt carry."""
    return {"schema": document["schema"], **document["counts"]}


def write(out: Path, match_id: str, destination: Path | None = None) -> dict:
    """``MISSED_PLAY.json`` of a product-runner match directory (upstream on disk)."""
    document = build(Path(out) / "upstream" / match_id, match_id)
    target = Path(destination or out) / FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=1) + "\n")
    temporary.replace(target)
    return document


def upstream_inputs(match_dir: Path, match_id: str, scratch: Path) -> Path:
    """The upstream folder this module reads: the tree, or the inputs from the match's
    ``archive/upstream.tar.zst`` extracted under ``scratch`` (a compacted match)."""
    import subprocess

    tree = Path(match_dir) / "upstream" / match_id
    archive = Path(match_dir) / "archive" / "upstream.tar.zst"
    if (tree / "shot_views_v1.csv").is_file() or not archive.is_file():
        return tree
    subprocess.run(
        [
            "tar",
            "-I",
            "zstd -d --long=30",
            "-xf",
            str(archive),
            "-C",
            str(scratch),
            "--wildcards",
            "--no-wildcards-match-slash",
            *(f"upstream/{match_id}/{name}" for name in INPUT_NAMES),
            f"upstream/{match_id}/run_manifests/*.json",
        ],
        check=False,
        capture_output=True,
    )
    extracted = Path(scratch) / "upstream" / match_id
    extracted.mkdir(parents=True, exist_ok=True)
    for name in (LEDGER_NAME, "source_integrity.json"):
        if (tree / name).is_file() and not (extracted / name).is_file():
            (extracted / name).write_bytes((tree / name).read_bytes())
    return extracted


def backfill(match_dir: Path, match_id: str, destination: Path) -> dict:
    """``MISSED_PLAY.json`` of a finished (possibly compacted, read-only) match into
    ``destination``."""
    import tempfile

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="missed_play_", dir=destination) as scratch:
        upstream = upstream_inputs(match_dir, match_id, Path(scratch))
        document = build(upstream, match_id)
    document["source_dir"] = str(match_dir)
    (destination / FILE).write_text(json.dumps(document, indent=1) + "\n")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("upstream", type=Path, help="upstream/<match_id> folder")
    parser.add_argument("--out", type=Path, help="write the record here (default: print counts)")
    args = parser.parse_args()
    document = build(args.upstream)
    if args.out:
        args.out.write_text(json.dumps(document, indent=1) + "\n")
    print(json.dumps(document["counts"], indent=1))


if __name__ == "__main__":
    main()
