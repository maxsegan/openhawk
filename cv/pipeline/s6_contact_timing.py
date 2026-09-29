"""Bounded source-PMF timing alternatives for existing optional contacts.

The nearest-event time head does not establish contact occurrence. Original
occurrence hypotheses stay literal alternatives; this policy only proposes
other preparation epochs, before physical fitting or evaluation gates.
"""

from __future__ import annotations

from copy import deepcopy
import numpy as np

from cv.pipeline import event_time_distribution as timing

POLICY = "pmf_peaks"
SCHEMA = "s6_contact_timing_pmf_v1"
MAX_ADDITIONAL = 2
SUPPRESSION_FRAMES = 2.0
SEARCH_RADIUS = 1.0
MIN_MASS = 1.0 / timing.TIME_BINS


def policy(document: dict) -> str:
    value = document.get("contact_timing", {}).get("policy", "off")
    if value not in {"off", POLICY}:
        raise ValueError("unknown optional contact timing policy")
    return value


def _rows(source: dict) -> tuple[float, list[dict]] | None:
    block = source.get("time_neighborhood")
    if block is None:
        return None
    from cv.pipeline import event_time_neighborhood

    anchor = float(source["candidate_frame"])
    event_time_neighborhood.validate(
        block,
        clip=source["clip"],
        candidate_frame=anchor,
        classes=("none", "contact", "bounce", "net_hit"),
    )
    return anchor, block["rows"]


def prepare(
    candidate: dict,
    attempt: dict,
    gap: dict,
    pts: dict[int, float],
    cuts: set[int],
    unsupported: set[int],
) -> dict:
    """Derive candidates once from ordinary source evidence, never fitted results."""
    source = _rows(candidate["source_emission"])
    receipt = {
        "schema": SCHEMA,
        "policy": POLICY,
        "alternatives": [],
        "excluded_peaks": [],
        "native_timestamps_changed": False,
        "occurrence_support_reused_not_retimed": True,
    }
    if source is None:
        return {**receipt, "status": "unavailable_in_source_emission"}
    anchor, rows = source
    frames = sorted(pts)
    if anchor not in pts:
        return {**receipt, "status": "anchor_outside_native_scope"}
    # A cut begins a new native segment. Unsupported camera/view frames also
    # bound the domain; no neighbor may reach through one to vote on a peak.
    barriers = set(cuts) | unsupported
    left = max(
        [frames[0], *[f for f in cuts if f <= anchor], *[f + 1 for f in unsupported if f < anchor]]
    )
    right = min([frames[-1], *[f - 1 for f in barriers if f > anchor]])
    low = max(float(left), anchor - timing.TIME_RADIUS, float(gap["interval"][0]))
    high = min(float(right), anchor + timing.TIME_RADIUS, float(gap["interval"][1]))
    receipt["admissible_native_domain"] = [low, high]
    receipt["source_neighborhood_anchor"] = anchor
    receipt["source_rows_used"] = []
    if anchor in unsupported or low >= high:
        return {**receipt, "status": "no_contiguous_supported_native_domain"}
    envelope: dict[float, dict] = {}
    for row in rows:
        centre = float(row["candidate_frame"])
        if not left <= centre <= right:
            continue
        receipt["source_rows_used"].append(row["source_row_index"])
        distribution = row["time_distribution"]
        for offset, mass in zip(
            distribution["offset_frames"], distribution["probabilities"], strict=True
        ):
            epoch = centre + float(offset)
            if not low <= epoch <= high or epoch not in pts:
                continue
            old = envelope.get(epoch)
            reading = {
                "epoch": epoch,
                "mass": float(mass),
                "source_row_index": row["source_row_index"],
                "candidate_frame": centre,
                "offset_frames": float(offset),
            }
            if old is None or (-reading["mass"], centre) < (-old["mass"], old["candidate_frame"]):
                envelope[epoch] = reading
    receipt["max_envelope"] = [envelope[e] for e in sorted(envelope)]
    maxima = [
        r
        for e, r in envelope.items()
        if r["mass"] >= MIN_MASS
        and r["mass"] >= envelope.get(e - 1, {"mass": -1})["mass"]
        and r["mass"] >= envelope.get(e + 1, {"mass": -1})["mass"]
        and (
            r["mass"] > envelope.get(e - 1, {"mass": -1})["mass"]
            or r["mass"] > envelope.get(e + 1, {"mass": -1})["mass"]
        )
    ]
    selected = [float(candidate["event"]["frame"])]
    for reading in sorted(maxima, key=lambda r: (-r["mass"], r["epoch"])):
        epoch = reading["epoch"]
        if any(abs(epoch - other) <= SUPPRESSION_FRAMES for other in selected):
            continue
        a, b = max(low, epoch - SEARCH_RADIUS), min(high, epoch + SEARCH_RADIUS)
        reason = None
        if not float(gap["interval"][0]) < a < epoch < b < float(gap["interval"][1]):
            reason = "no_open_source_search_interval"
        elif any(a <= f <= b for f in barriers):
            reason = "source_interval_crosses_view_or_camera_barrier"
        elif gap.get("kind", "interior") == "interior" and not (
            any(e["frame"] < a for e in gap["bounces"])
            and any(e["frame"] > b for e in gap["bounces"])
        ):
            reason = "contact_no_longer_separates_source_grounds"
        elif any(a <= float(e["frame"]) <= b for e in attempt["events"]):
            reason = "source_interval_overlaps_accepted_event"
        elif any(
            (float(e["frame"]) - epoch) * (float(e["frame"]) - float(candidate["event"]["frame"]))
            <= 0
            for e in attempt["events"]
        ):
            reason = "timing_alternative_reorders_accepted_event"
        elif gap.get("kind") == "final" and sum(e["frame"] < a for e in gap["bounces"]) >= 2:
            reason = "two_source_grounds_before_contact"
        event = deepcopy(candidate["event"])
        event.update(
            frame=epoch, frame_interval=[a, b], interval_origin=SCHEMA, epoch_source=SCHEMA
        )
        # Retain old model location as original evidence, not as an epoch-local
        # location for the new event. Scene player/camera context is reprepared.
        original = deepcopy(candidate["event"])
        for key in ("automatic_location", "emitted_rounded_frame"):
            event.pop(key, None)
        event["timing_alternative"] = {
            "schema": SCHEMA,
            "source_emission_index": candidate["source_emission_index"],
            "original_event": original,
            "peak": deepcopy(reading),
        }
        if reason is None and attempt.get("observed_horizon_tail") is not None:
            from cv.experiments.connected_shooting import observed_horizon_tail

            try:
                observed_horizon_tail.for_events(
                    attempt["observed_horizon_tail"], [*attempt["events"], event]
                )
            except (ValueError, KeyError, IndexError):
                reason = "no_original_native_suffix_support"
        if reason:
            receipt["excluded_peaks"].append({**reading, "reason": reason})
            continue
        selected.append(epoch)
        receipt["alternatives"].append(
            {
                "event": event,
                "source_pts_interval": np.interp([a, b], frames, [pts[f] for f in frames]).tolist(),
                "peak": deepcopy(reading),
            }
        )
        if len(receipt["alternatives"]) == MAX_ADDITIONAL:
            break
    receipt["status"] = "prepared" if receipt["alternatives"] else "no_distinct_admissible_peak"
    return receipt


def extend_hypotheses(original: list[dict], document: dict) -> list[dict]:
    """Append a bounded source-ranked lane; never evict original alternatives."""
    if policy(document) == "off":
        return original
    by_id = {r["id"]: r for r in document["candidates"]}
    choices = []
    for base_index, hypothesis in enumerate(original):
        for candidate_id in hypothesis["candidate_ids"]:
            row = by_id[candidate_id]
            for alternative_index, alternative in enumerate(
                row.get("contact_timing", {}).get("alternatives", [])
            ):
                choices.append(
                    (
                        -alternative["peak"]["mass"],
                        base_index,
                        candidate_id,
                        alternative_index,
                        alternative,
                    )
                )
    result = list(original)
    for _, base_index, candidate_id, index, alternative in sorted(choices, key=lambda r: r[:4])[
        :MAX_ADDITIONAL
    ]:
        h = deepcopy(original[base_index])
        old = next(
            e
            for e in h["added"]
            if e["optional_topology_membership"]["candidate_id"] == candidate_id
        )
        event = deepcopy(alternative["event"])
        event["occurrence_status"] = old["occurrence_status"]
        event["optional_topology_membership"] = deepcopy(old["optional_topology_membership"])
        if "optional_gap_kind" in old:
            event["optional_gap_kind"] = old["optional_gap_kind"]
        h["events"] = sorted(
            [event if e == old else e for e in h["events"]], key=lambda e: e["frame"]
        )
        h["added"] = [event if e == old else e for e in h["added"]]
        h["name"] += f"_pmf_{candidate_id}_{index + 1}"
        h["contact_timing_alternative"] = {
            "policy": POLICY,
            "base_hypothesis": original[base_index]["name"],
            "candidate_id": candidate_id,
            "peak": deepcopy(alternative["peak"]),
        }
        result.append(h)
    return result
