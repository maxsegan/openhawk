"""Fit-only bounce rules v1 applied to a 3D viewer point document (``point3d_v1``).

One transform shared by the post-hoc re-tier of published matches
(``cv/experiments/retier_fit_only``) and the native portal step: given a document and the
rule decisions of its attempt's validated flights (``fit_only_bounce_rules.evaluate_result``),

* rule 1: every fitted state of an accepted flight past its ``end_frame`` (+``CONTINUATION_EPS``)
  is marked ``frames.acceptance = "modelled_continuation"`` (kept, drawn grey and labelled by
  the viewer, never validated), bounces there get ``modelled_continuation: true``, and the
  overlay video's fitted path stops at the validated end;
* rules 2/3: a demoted flight's ``per_flight`` row is no longer accepted (``tier:
  "best_effort"``, ``failures`` = the rule reasons), its states are marked
  ``acceptance = "best_effort"`` (the viewer's best-effort styling, not the metric tube), its
  overlay-video marker moves from ``fit`` to ``rejected`` (the overlay's "failed its gates"
  layer) and the point is no longer ``complete_point``.

Nothing else changes; the document gets a ``fit_only_bounce_rules`` block naming the version,
the demoted flights and the counts. Idempotent: a document that already carries the block of
the same version is returned unchanged.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from cv.pipeline import fit_only_bounce_rules as rules

BEST_EFFORT = "best_effort"
NOTE = "re-tiered by fit-only-bounce rules v1"
#: ``portal_flight_overlay.PATH_POINTS``: the overlay path is the flight's projected states,
#: evenly subsampled to at most this many points.
PATH_POINTS = 32


def _segment_index(segment: Any) -> int | None:
    match = re.fullmatch(r"flight_(\d+)", str(segment or ""))
    return int(match.group(1)) if match else None


def _key(start: float) -> float:
    return round(float(start), 4)


def _cut_path(path: list, frames: list[int], end: float) -> list:
    """The overlay path's points up to ``end`` (``frames``: the flight's state frames, sorted)."""
    n = len(frames)
    if not path or not n:
        return path
    if n <= PATH_POINTS and len(path) == n:
        sources = frames
    elif len(path) == PATH_POINTS:
        sources = [frames[round(i * (n - 1) / (PATH_POINTS - 1))] for i in range(PATH_POINTS)]
    else:
        return path  # some projections failed: the mapping is unknown, keep the path
    return [p for p, f in zip(path, sources, strict=True) if f <= end + rules.CONTINUATION_EPS]


def flight_states(document: dict[str, Any]) -> dict[float, list[int]]:
    """Flight start frame (rounded to 4) -> sorted native frames with a fitted state."""
    frames = document.get("frames") or {}
    rows = {
        r.get("flight_index"): r for r in (document.get("per_flight") or {}).get("flights") or []
    }
    out: dict[float, list[int]] = {}
    xs = frames.get("x") or []
    for i, segment in enumerate(frames.get("segment") or []):
        row = rows.get(_segment_index(segment))
        if row is not None and i < len(xs) and xs[i] is not None:
            out.setdefault(_key(row["start_frame"]), []).append(int(frames["frame"][i]))
    return {k: sorted(v) for k, v in out.items()}


def label_document(
    document: dict[str, Any],
    demoted: dict[float, list[str]],
    states: dict[float, list[int]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(new document, change summary). ``demoted``: flight start frame (rounded to 4) ->
    rule reasons, for this attempt's validated flights the rules demote. ``states``: flight
    start -> state frames over all the attempt's documents (``flight_states``), for an attempt
    document whose overlay covers flights fitted in its component documents."""
    block = document.get("fit_only_bounce_rules") or {}
    if block.get("rules_version") == rules.RULES_VERSION:
        return document, {"changed": False, "already": True}
    doc = copy.deepcopy(document)
    per_flight = doc.get("per_flight") or {}
    rows = per_flight.get("flights") or []
    by_local = {row.get("flight_index"): row for row in rows}
    demoted_rows = []
    for row in rows:
        reasons = demoted.get(_key(row.get("start_frame", -1)))
        if reasons and row.get("accepted"):
            row["accepted"] = False
            row["tier"] = BEST_EFFORT
            row["failures"] = list(row.get("failures") or []) + list(reasons)
            row["fit_only_bounce_rules"] = {
                "rules_version": rules.RULES_VERSION,
                "reasons": reasons,
            }
            checks = [r for r in reasons if str(r).startswith("post_fit_")]
            if checks:
                # Demoted by the post-fit checks (``post_fit_checks``), same best-effort path.
                row["post_fit_checks"] = {"reasons": checks}
            demoted_rows.append(row)
    demoted_local = {row["flight_index"] for row in demoted_rows}
    if demoted_rows:
        per_flight["accepted_flight_indices"] = [
            i for i in per_flight.get("accepted_flight_indices") or [] if i not in demoted_local
        ]
        per_flight["best_effort_flight_indices"] = sorted(demoted_local)
        if "accepted_flight_count" in per_flight:
            per_flight["accepted_flight_count"] = len(per_flight["accepted_flight_indices"])
        if per_flight.get("complete_point"):
            per_flight["complete_point"] = False
        if per_flight.get("complete_real_point_accepted"):
            per_flight["complete_real_point_accepted"] = False
    frames = doc.get("frames") or {}
    acceptance = frames.get("acceptance")
    continuation_frames = best_effort_frames = 0
    if isinstance(acceptance, list):
        for i, segment in enumerate(frames.get("segment") or []):
            local = _segment_index(segment)
            row = by_local.get(local)
            if row is None:
                continue
            if acceptance[i] != "accepted":
                continue
            if float(frames["frame"][i]) > float(row["end_frame"]) + rules.CONTINUATION_EPS:
                acceptance[i] = rules.CONTINUATION
                continuation_frames += 1
            elif local in demoted_local:
                acceptance[i] = BEST_EFFORT
                best_effort_frames += 1
    continuation_bounces = 0
    for bounce in doc.get("bounces") or []:
        row = by_local.get(_segment_index(bounce.get("segment")))
        if row is None:
            continue
        if float(bounce.get("frame", -1)) > float(row["end_frame"]) + rules.CONTINUATION_EPS:
            if row.get("accepted") or row["flight_index"] in demoted_local:
                bounce[rules.CONTINUATION] = True
                continuation_bounces += 1
        elif row["flight_index"] in demoted_local:
            bounce["tier"] = BEST_EFFORT
    overlay = doc.get("flight_overlay") or {}
    overlay_cut = overlay_moved = 0
    all_states = {**(states or {}), **flight_states(doc)}
    spans, accepted_flights = [], []
    for flight in overlay.get("flights") or []:
        if not flight.get("accepted"):
            continue
        reasons = demoted.get(_key(flight.get("start_frame", -1)))
        if reasons:
            flight["accepted"] = False
            flight["tier"] = BEST_EFFORT
            flight["failed_checks"] = list(flight.get("failed_checks") or []) + list(reasons)
            spans.append((float(flight["start_frame"]), float(flight["end_frame"])))
        else:
            accepted_flights.append(flight)
    for record in overlay.get("frames") or []:
        frame = record.get("frame")
        if "fit" in record or "fit_path" in record:
            if any(lo <= frame <= hi for lo, hi in spans) and not _accepted_at(
                frame, accepted_flights
            ):
                for name in ("fit", "fit_path"):
                    if name in record:
                        record[name.replace("fit", "rejected")] = record.pop(name)
                overlay_moved += 1
        if "fit_path" in record:
            flight = _row_at(frame, accepted_flights)
            if flight is not None:
                cut = _cut_path(
                    record["fit_path"],
                    all_states.get(_key(flight["start_frame"]), []),
                    float(flight["end_frame"]),
                )
                if len(cut) != len(record["fit_path"]):
                    overlay_cut += 1
                    if len(cut) >= 2:
                        record["fit_path"] = cut
                    else:
                        record.pop("fit_path")
    overlay_demoted = len(spans)
    summary = {
        "changed": bool(
            demoted_rows
            or continuation_frames
            or continuation_bounces
            or overlay_cut
            or overlay_moved
            or overlay_demoted
        ),
        "demoted_flights": [
            {
                "flight_index": r["flight_index"],
                "start_frame": r["start_frame"],
                "end_frame": r["end_frame"],
                "reasons": r["fit_only_bounce_rules"]["reasons"],
            }
            for r in demoted_rows
        ],
        "continuation_frames": continuation_frames,
        "continuation_bounces": continuation_bounces,
        "best_effort_frames": best_effort_frames,
        "overlay_paths_cut": overlay_cut,
        "overlay_markers_moved": overlay_moved,
        "overlay_flights_demoted": overlay_demoted,
    }
    if not summary["changed"]:
        return document, summary
    doc["fit_only_bounce_rules"] = {
        "rules_version": rules.RULES_VERSION,
        "note": NOTE,
        **{k: v for k, v in summary.items() if k != "changed"},
    }
    return doc, summary


def _row_at(frame: float, rows: list[dict]) -> dict | None:
    """The accepted flight whose overlay marker this frame shows (``portal_flight_overlay``)."""
    inside = [r for r in rows if float(r["start_frame"]) <= frame <= float(r["end_frame"])]
    near = inside or [
        r for r in rows if float(r["start_frame"]) - 2 <= frame <= float(r["end_frame"]) + 2
    ]
    return max(near, key=lambda r: float(r["start_frame"])) if near else None


def _accepted_at(frame: float, rows: list[dict]) -> bool:
    return any(float(r["start_frame"]) <= frame <= float(r["end_frame"]) for r in rows)
