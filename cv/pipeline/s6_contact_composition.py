"""Bounded two-contact composition inside one original interior source gap.

Optional engineering ablation, default off. A gap bounded by two accepted
contacts that already carries two source grounds can require two returns, but
the existing selector proposes at most one addition per gap, so two genuinely
co-occurring contacts are never a literal alternative. This helper only decides
whether two already-witnessed source candidates are compatible with the
original record: no raw input, label, timing choice or fit is introduced, and
the ranking stays the source-evidence order the single-addition beam uses.

Nothing here fits, gates or evaluates. Every refusal is an explicit receipt.
"""

from __future__ import annotations

import itertools
import math

from cv.pipeline import s6_optional_contacts as optional

POLICY = "bounded_pairs"
OFF = "off"
SCHEMA = "s6_contact_composition_v1"
# Source pool per gap, compatible pairs examined per gap, and the original
# native observation floor each new subgap must meet. The floor is an adequacy
# statement about the observed record, not a claim about fitted truth.
MAX_POOL = 3
MAX_PAIRS = 3
MIN_OBSERVATIONS = 4
MAX_ADDED_PER_GAP = 2
MAX_ALTERNATIVES = 2
BLOCKING_EVENTS = ("contact", "bounce", "net_hit")
VISIBLE = "visible"
DERIVED_SUPPORT = "interpolated_estimate"


def policy(document: dict) -> str:
    """The witness's own composition declaration; absence is off, not a default."""
    if "contact_composition" not in document:
        return OFF
    declaration = document["contact_composition"]
    if not isinstance(declaration, dict) or declaration.get("policy") not in {OFF, POLICY}:
        raise ValueError("unknown optional contact composition declaration")
    return declaration["policy"]


def _interval(event: dict) -> tuple[float, float]:
    """The shared event-interval convention; a point event is its own interval."""
    a, b = event.get("frame_interval", [event["frame"], event["frame"]])
    return float(a), float(b)


def _rank_key(row: dict) -> tuple:
    # Identical ordering to ``s6_optional_contacts._ranked``; only the cap differs.
    return (-row["witness_support"], -row["occurrence_log_odds"], row["event"]["frame"], row["id"])


def ranked_pool(rows: list[dict]) -> list[dict]:
    """Top source candidates of one gap under the unchanged single-addition sort."""
    return sorted(rows, key=_rank_key)[:MAX_POOL]


def _branch_key(rows: tuple[dict, ...]) -> tuple:
    # The unchanged beam order of the single-addition selector: source evidence
    # only, no fitted residual and no gate verdict.
    return (
        -sum(r["witness_support"] for r in rows),
        -sum(r["occurrence_log_odds"] for r in rows),
        tuple(r["id"] for r in rows),
    )


def admitted_observations(observations: list[dict]) -> tuple[list[int], dict]:
    """Original native visible ball frames; interpolated estimates are not observations.

    ``observations`` is the source-bound owner observation record. A derived or
    non-finite row is dropped with a reason rather than counted, and one native
    frame contributes at most one observation.
    """
    points: dict[int, tuple[float, float]] = {}
    conflicting: set[int] = set()
    excluded: list[dict] = []

    def drop(frame, reason):
        excluded.append({"frame": frame, "reason": reason})

    for row in observations:
        frame = row.get("frame")
        if row.get("status") != VISIBLE:
            drop(frame, "status_is_not_visible")
            continue
        if row.get("support_class") == DERIVED_SUPPORT:
            drop(frame, "interpolated_estimate_is_not_an_observation")
            continue
        if (
            not isinstance(frame, (int, float))
            or isinstance(frame, bool)
            or not math.isfinite(frame)
            or float(frame) != int(frame)
        ):
            drop(frame, "non_integer_native_frame")
            continue
        frame = int(frame)
        try:
            point = (float(row["x1080"]), float(row["y1080"]))
        except (KeyError, TypeError, ValueError):
            drop(frame, "unreadable_native_point")
            continue
        if not all(math.isfinite(value) for value in point):
            drop(frame, "non_finite_native_point")
            continue
        if frame in conflicting:
            drop(frame, "conflicting_duplicate_native_frame")
            continue
        old = points.get(frame)
        if old is not None and old != point:
            # Two different points for one native frame do not establish one
            # observed location; neither row may support a subgap.
            points.pop(frame)
            conflicting.add(frame)
            drop(frame, "conflicting_duplicate_native_frame")
            continue
        points[frame] = point
    return sorted(points), {
        "source": "original_native_visible_ball_observations",
        "admitted_frames": len(points),
        "excluded_rows": len(excluded),
        "excluded": excluded,
        "interpolated_or_derived_used": False,
    }


def gap_boundaries(document: dict, events: list[dict], gap_index: int) -> tuple[dict | None, str]:
    """The actual original contact intervals bounding one prepared interior gap.

    Endpoints are the supplied contacts themselves. Nothing is fabricated: an
    unmatched or ambiguous boundary refuses the gap instead of inventing one.
    """
    gaps = document.get("gaps") or []
    if not 0 <= gap_index < len(gaps):
        return None, "gap_index_outside_the_witness"
    gap = gaps[gap_index]
    if gap.get("kind", optional.INTERIOR) != optional.INTERIOR:
        return None, "gap_is_not_an_original_interior_gap"
    low, high = (float(x) for x in gap["interval"])
    bounding = []
    for edge in (low, high):
        found = [
            e for e in events if e.get("event_type") == "contact" and float(e["frame"]) == edge
        ]
        if len(found) != 1:
            return None, "gap_boundary_is_not_one_supplied_contact"
        bounding.append(_interval(found[0]))
    return {
        "low": low,
        "high": high,
        "left_interval": list(bounding[0]),
        "right_interval": list(bounding[1]),
    }, ""


def qualify_pair(
    items: list[tuple[str, dict]],
    boundaries: dict,
    events: list[dict],
    frames: list[int],
    *,
    gap_index: int,
) -> dict:
    """One explicit compatibility receipt for two added contacts in one gap.

    ``items`` carry the ACTUAL proposed events, so a retimed alternative is
    rechecked against its siblings rather than assumed to keep its preparation
    geometry. A subgap with no supplied ground is an ordinary volley and is
    recorded, never refused; two grounds inside one new subgap are refused.
    """
    receipt = {
        "gap_index": gap_index,
        "candidate_ids": [candidate_id for candidate_id, _ in items],
        "eligible": False,
        "reason": None,
        "minimum_observations": MIN_OBSERVATIONS,
        "subgaps": [],
    }

    def refuse(reason: str) -> dict:
        receipt["reason"] = reason
        return receipt

    if len({candidate_id for candidate_id, _ in items}) != len(items):
        return refuse("repeated_candidate_id")
    if len(items) != MAX_ADDED_PER_GAP:
        return refuse("composition_qualifies_exactly_two_added_contacts")
    measured = []
    for candidate_id, event in items:
        a, b = _interval(event)
        frame = float(event.get("frame", a))
        if not all(math.isfinite(x) for x in (a, b, frame)) or not a <= frame <= b:
            return refuse("non_finite_or_unordered_candidate_interval")
        measured.append((candidate_id, a, b, frame))
    ordered = sorted(measured, key=lambda row: (row[1], row[2], row[0]))
    receipt["candidate_ids"] = [row[0] for row in ordered]
    receipt["intervals"] = [[row[1], row[2]] for row in ordered]
    (_, a1, b1, f1), (_, a2, b2, f2) = ordered
    if not b1 < a2:
        return refuse("pair_intervals_overlap_or_are_not_strictly_ordered")
    if not f1 < f2:
        return refuse("ambiguous_pair_epoch_order")
    if not (boundaries["low"] < a1 and b2 < boundaries["high"]):
        return refuse("pair_is_not_wholly_inside_the_original_gap")
    for event in events:
        if event.get("event_type") not in BLOCKING_EVENTS:
            continue
        start, end = _interval(event)
        if any(start <= b and a <= end for a, b in ((a1, b1), (a2, b2))):
            return refuse("pair_overlaps_a_supplied_event_interval")
    edges = (
        ("left_contact_to_first", boundaries["left_interval"][1], a1),
        ("first_to_second", b1, a2),
        ("second_to_right_contact", b2, boundaries["right_interval"][0]),
    )
    bounces = [
        e
        for e in events
        if e.get("event_type") == "bounce"
        and boundaries["low"] < float(e["frame"]) < boundaries["high"]
    ]
    for name, low, high in edges:
        # Observations strictly outside every contact interval, and supplied
        # grounds kept at their own unchanged epochs.
        observed = [f for f in frames if low < f < high]
        inside = [float(e["frame"]) for e in bounces if low < float(e["frame"]) < high]
        receipt["subgaps"].append(
            {
                "name": name,
                "interval": [low, high],
                "observation_frames": observed,
                "observation_count": len(observed),
                "supplied_bounce_frames": inside,
                "supplied_bounce_count": len(inside),
            }
        )
    if any(row["supplied_bounce_count"] > 1 for row in receipt["subgaps"]):
        return refuse("more_than_one_supplied_bounce_in_one_new_subgap")
    if sum(row["supplied_bounce_count"] for row in receipt["subgaps"]) != len(bounces):
        return refuse("supplied_bounce_lies_in_no_new_subgap")
    if any(row["observation_count"] < MIN_OBSERVATIONS for row in receipt["subgaps"]):
        return refuse("new_subgap_below_the_original_observation_floor")
    receipt["eligible"] = True
    return receipt


def _advance(
    legacy: tuple[dict, ...] | None,
    paired: tuple[dict, ...] | None,
    pool: list[dict],
    pairs: list[tuple[dict, ...]],
) -> tuple[tuple[dict, ...] | None, tuple[dict, ...] | None]:
    """One bounded dynamic-programming step: best no-pair and best paired prefix.

    Two retained states per gap, never the powerset of gap compositions.
    """
    singles = [(row,) for row in pool]
    extensions = []
    if legacy is not None:
        extensions += [(*legacy, *pair) for pair in pairs]
    if paired is not None:
        extensions += [(*paired, *option) for option in singles + pairs]
    return (
        min(((*legacy, *option) for option in singles), key=_branch_key)
        if legacy is not None
        else None,
        min(extensions, key=_branch_key) if extensions else None,
    )


def extend_interior_beam(
    document: dict,
    events: list[dict],
    legacy_beam: list[tuple[dict, ...]],
    groups: dict[int, list[dict]],
    interior: list[int],
    observations: list[dict],
) -> tuple[list[tuple[dict, ...]], dict | None]:
    """Offer one paired-interior branch beside the existing single-addition beam.

    Returns the original beam object unchanged, with no receipt, while the
    policy is off; the observation record is not read at all in that case. When
    on, inputs are never mutated, the existing source-best branch stays the
    first alternative, and at most one additional branch is proposed.
    """
    if policy(document) == OFF:
        return legacy_beam, None
    receipt = {
        "schema": SCHEMA,
        "policy": POLICY,
        "max_pool": MAX_POOL,
        "max_pairs_per_gap": MAX_PAIRS,
        "minimum_subgap_observations": MIN_OBSERVATIONS,
        "maximum_alternatives": MAX_ALTERNATIVES,
        "pools": [],
        "pairs": [],
        "chosen_paired_candidate_ids": [],
        "source_evidence_rank_only": True,
        "fitted_or_gated": False,
        "runtime_model_calls": 0,
    }
    frames, receipt["observations"] = admitted_observations(observations)
    legacy: tuple[dict, ...] | None = ()
    paired: tuple[dict, ...] | None = None
    for gap_index in interior:
        supported = groups.get(gap_index) or []
        pool = ranked_pool(supported)
        receipt["pools"].append(
            {
                "gap_index": gap_index,
                "supported_candidates": len(supported),
                "pool_candidate_ids": [row["id"] for row in pool],
            }
        )
        if not pool:
            # Every original interior gap must keep its support, exactly as the
            # single-addition selector requires.
            receipt["status"] = "interior_gap_without_supported_candidate"
            return legacy_beam, receipt
        boundaries, reason = gap_boundaries(document, events, gap_index)
        pairs: list[tuple[dict, ...]] = []
        if boundaries is None:
            receipt["pairs"].append(
                {"gap_index": gap_index, "candidate_ids": [], "eligible": False, "reason": reason}
            )
        else:
            for first, second in list(itertools.combinations(pool, 2))[:MAX_PAIRS]:
                by_id = {first["id"]: first, second["id"]: second}
                qualified = qualify_pair(
                    [(first["id"], first["event"]), (second["id"], second["event"])],
                    boundaries,
                    events,
                    frames,
                    gap_index=gap_index,
                )
                receipt["pairs"].append(qualified)
                if qualified["eligible"]:
                    pairs.append(tuple(by_id[i] for i in qualified["candidate_ids"]))
        legacy, paired = _advance(legacy, paired, pool, pairs)
    if paired is None:
        receipt["status"] = "no_eligible_pair"
        return legacy_beam, receipt
    if not legacy_beam or not legacy_beam[0]:
        receipt["status"] = "no_original_branch_to_accompany"
        return legacy_beam, receipt
    chosen = [row["id"] for row in paired]
    if chosen == [row["id"] for row in legacy_beam[0]]:
        receipt["status"] = "paired_branch_duplicates_the_original_branch"
        return legacy_beam, receipt
    receipt.update(
        status="paired_alternative_added",
        chosen_paired_candidate_ids=chosen,
        original_alternative_candidate_ids=[row["id"] for row in legacy_beam[0]],
    )
    return [legacy_beam[0], paired], receipt


def admissibility(
    hypothesis: dict, document: dict, events: list[dict], observations: list[dict]
) -> dict:
    """Recheck any two added contacts of one gap against their ACTUAL events.

    Timing expansion may retime one member of a pair, so compatibility is
    recomputed from the hypothesis rather than reused from preparation; the
    returned receipt also refreshes the observed support of the retimed pair.
    Identity comes from the immutable witness candidates. A single addition per
    gap is the original arrangement and stays eligible. No gate is consulted.
    """
    receipt = {
        "schema": SCHEMA,
        "policy": policy(document),
        "eligible": True,
        "pairs": [],
        "checked_gaps": [],
        "candidate_ids": [],
    }
    if receipt["policy"] == OFF:
        return {**receipt, "status": "policy_off"}
    by_id = {row["id"]: row for row in document.get("candidates", [])}
    grouped: dict[int, list[tuple[str, dict]]] = {}
    for event in hypothesis.get("added", []):
        membership = event.get("optional_topology_membership") or {}
        candidate_id = membership.get("candidate_id")
        if candidate_id not in by_id:
            return {
                **receipt,
                "eligible": False,
                "status": "unknown_added_candidate_id",
                "candidate_id": candidate_id,
            }
        if candidate_id in receipt["candidate_ids"]:
            return {
                **receipt,
                "eligible": False,
                "status": "repeated_added_candidate_id",
                "candidate_id": candidate_id,
            }
        receipt["candidate_ids"].append(candidate_id)
        grouped.setdefault(int(by_id[candidate_id]["gap_index"]), []).append((candidate_id, event))
    frames = None
    for gap_index, items in sorted(grouped.items()):
        if len(items) < 2:
            continue
        receipt["checked_gaps"].append(gap_index)
        ids = [candidate_id for candidate_id, _ in items]
        if len(items) > MAX_ADDED_PER_GAP:
            receipt["eligible"] = False
            receipt["pairs"].append(
                {
                    "gap_index": gap_index,
                    "candidate_ids": ids,
                    "eligible": False,
                    "reason": "more_than_two_added_contacts_in_one_gap",
                }
            )
            continue
        if frames is None:
            frames, receipt["observations"] = admitted_observations(observations)
        boundaries, reason = gap_boundaries(document, events, gap_index)
        if boundaries is None:
            receipt["eligible"] = False
            receipt["pairs"].append(
                {
                    "gap_index": gap_index,
                    "candidate_ids": ids,
                    "eligible": False,
                    "reason": reason,
                }
            )
            continue
        qualified = qualify_pair(items, boundaries, events, frames, gap_index=gap_index)
        receipt["pairs"].append(qualified)
        receipt["eligible"] = receipt["eligible"] and qualified["eligible"]
    receipt["status"] = "rechecked" if receipt["pairs"] else "at_most_one_addition_per_gap"
    return receipt
