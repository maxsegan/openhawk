"""Flight-origin matcher used for the reported yield numbers (evaluation only).

A reference is one independently labelled competitive racket-contact origin with an inclusive
native-PTS interval ``origin_pts_ms`` and, when known, an endpoint interval ``endpoint_pts_ms``
(the next owned contact or a confirmed ending). An accepted model flight matches a reference
when its start time lies inside the origin interval widened by two native frame periods, and
the match is one-to-one: the reference has exactly one eligible flight and that flight is
eligible for no other reference. Abstentions and missing output keep every reference in the
denominator. Model output is never used to build references.

This file is a faithful extraction of the two functions the private evaluation harness calls
(``match_attempt`` and ``interpolate_native_pts_ms``; source files sha256 4851838908df... and
045fdbc462843a...), with the file-loading plumbing removed. The harness around it (reference
building from labels, digest-pinned bindings, component aggregation and the post-ending /
origin-offset classification of unmatched accepts) is not part of this release.

Inputs:
- ``references``: [{"id", "origin_pts_ms": [lo, hi], "endpoint_pts_ms": [lo, hi] | None,
  "competitive_confirmed": bool}, ...]
- ``result``: an S6 ``result.json`` (``verdict.flights`` with ``accepted``, ``start_frame``,
  ``end_frame`` in fractional clip frames)
- ``images``: the attempt's picture records [{"frame": int, "native_pts_seconds": float}, ...]
- ``native_hz``: native picture rate of the source
"""

from __future__ import annotations

import math
from collections import Counter


def interpolate_native_pts_ms(frame: float, images: list[dict]) -> tuple[float | None, str]:
    """Native PTS (ms) of a fractional clip frame; None with a reason when unbound.

    ``images`` are the attempt's own ``source_pack.images`` records with integer
    ``frame`` and ``native_pts_seconds``.  Interpolation is only between two
    pictures whose frame numbers are consecutive integers; a missing neighbour is
    a gap and yields no timestamp (no extrapolation).
    """
    if frame is None or not math.isfinite(frame):
        return None, "nonfinite_frame"
    by_frame = {}
    for img in images:
        if img.get("native_pts_seconds") is None:
            continue
        by_frame[int(img["frame"])] = float(img["native_pts_seconds"]) * 1000.0
    lo = math.floor(frame)
    if abs(frame - round(frame)) < 1e-9:
        pts = by_frame.get(int(round(frame)))
        return (pts, "exact") if pts is not None else (None, "frame_not_in_window")
    hi = lo + 1
    if lo not in by_frame or hi not in by_frame:
        return None, "gap_or_outside_window"
    a, b = by_frame[lo], by_frame[hi]
    if b <= a:
        return None, "nonincreasing_native_pts"
    return a + (frame - lo) * (b - a), "interpolated"


def match_attempt(references, result, images, native_hz):
    tolerance_ms = 2000.0 / native_hz
    flights = []
    for f in result.get("verdict", {}).get("flights", []):
        start, why = interpolate_native_pts_ms(f.get("start_frame"), images)
        end, end_why = interpolate_native_pts_ms(f.get("end_frame"), images)
        flights.append(
            dict(f, start_pts_ms=start, end_pts_ms=end, start_binding=why, end_binding=end_why)
        )
    eligible = {}
    for r in references:
        lo, hi = r["origin_pts_ms"]
        eligible[r["id"]] = [
            i
            for i, f in enumerate(flights)
            if f.get("accepted")
            and f["start_pts_ms"] is not None
            and lo - tolerance_ms <= f["start_pts_ms"] <= hi + tolerance_ms
        ]
    owners = Counter(i for indices in eligible.values() for i in indices)
    rows = []
    for r in references:
        candidates = eligible[r["id"]]
        matched = len(candidates) == 1 and owners[candidates[0]] == 1
        endpoint = r.get("endpoint_pts_ms")
        span = None
        emission = None
        if matched:
            emission = flights[candidates[0]]
            if endpoint is not None:
                end = emission["end_pts_ms"]
                span = (
                    end is not None
                    and endpoint[0] - tolerance_ms <= end <= endpoint[1] + tolerance_ms
                )
        rows.append(
            dict(
                reference=r,
                origin_gate_matched=matched,
                full_observed_span_gate=span if matched else False,
                emission=emission,
                eligible_emission_indices=candidates,
            )
        )
    confirmed = [r for r in rows if r["reference"]["competitive_confirmed"]]
    uncertain = [r for r in rows if not r["reference"]["competitive_confirmed"]]
    return dict(
        tolerance_ms=tolerance_ms,
        confirmed_denominator=len(confirmed),
        confirmed_origin_gate_matches=sum(r["origin_gate_matched"] for r in confirmed),
        confirmed_full_span_gates=sum(r["full_observed_span_gate"] is True for r in confirmed),
        confirmed_endpoints_unknown=sum(
            r["reference"].get("endpoint_pts_ms") is None for r in confirmed
        ),
        uncertain_denominator=len(uncertain),
        uncertain_origin_gate_matches=sum(r["origin_gate_matched"] for r in uncertain),
        model_flights=len(flights),
        model_gate_passes=sum(bool(f.get("accepted")) for f in flights),
        stage_complete_gate=bool(result.get("verdict", {}).get("complete_point")),
        visually_useful_flights=None,
        rows=rows,
        note=(
            "Origin gates, complete observed spans, and native visual usefulness are distinct. "
            "Missing output retains every reference."
        ),
    )
