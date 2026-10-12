"""Fit-only bounce rules v1 (owner-approved 2026-10-10): the ONE definition, shared by the
native pipeline step (``product_runner.step_assemble`` / ``step_portal``) and the post-hoc
re-tier of published matches (``cv/experiments/retier_fit_only``).

Why: the fit-only bounce audit (``cv/experiments/fit_only_bounce_audit/REPORT.md``) found that
validated flights whose bounce the fitter modelled without any detector record ("fit-only")
were wrong 60/100 (matched detector-recorded controls 6/100). The rules below come from that
sample (post hoc); they are frozen here and checked on a fresh panel before becoming default.

Definitions (frozen; ``RULES_VERSION`` changes if any of them changes):

* Population: each validated flight (verdict ``accepted`` and post-validation ``confirmed``) of
  an attempt's final S6 result.
* Bounce of a flight: a modelled ground contact of the flight's dense fitted state, paired
  (:func:`pair`) with the verdict flight's ``bounce_witness`` supplied frames.
* FIT-ONLY: no paired supplied frame AND no fit-packet bounce event within ``JOIN_FRAMES``
  (packet events: the component source's ``original_events``, or the first-fit stage's
  ``inputs/packet.json``; refit and beam stages supply none). This is
  ``event_physics_labels.labels`` ``fit_route == "modelled_only"``.
* INSIDE: fitted frame < the verdict flight's ``end_frame``; otherwise AFTER (the fitted
  continuation drawn past the validated end).
* WITNESS: no detector record of another physical type (contact, net_hit; any status,
  including abstentions) within ``JOIN_FRAMES`` of the fitted frame (record epoch =
  ``location.frame_subpixel`` else ``frame``), AND the detector's bounce-class probability
  reaches ``WITNESS_P`` on a row round(f)-1..round(f)+1.

Rules:

1. Continuation: states of a flight past its verdict ``end_frame`` (+``CONTINUATION_EPS``) and
   AFTER bounces are a modelled continuation: never validated, never counted (the native
   step cuts them from the validated record; published documents label them
   ``modelled_continuation``).
2. An INSIDE fit-only bounce with an EMITTED (non-abstained) net_hit record within
   ``JOIN_FRAMES`` demotes the flight to the best-effort tier (``fit_only_bounce_at_net_hit``).
   Checked before rule 3.
3. An INSIDE fit-only bounce without a WITNESS demotes the flight to the best-effort tier
   (``fit_only_bounce_unwitnessed``). Every inside fit-only bounce counts.

Detector-supplied bounces next to an emitted net hit are a diagnostic only, never demoted.

Switch ``TENNIS_FIT_ONLY_BOUNCE_RULES`` (``off`` default = today's output, the rollback; ``on``).
"""

from __future__ import annotations

import io
import json
import os
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import numpy as np

RULES_VERSION = "fit_only_bounce_rules_v1"
SWITCH = "TENNIS_FIT_ONLY_BOUNCE_RULES"
#: Detector record or packet event joined to a fitted epoch: |difference| <= this (frames).
JOIN_FRAMES = 1.0
#: Supplied and modelled bounce epochs pair when counts differ: nearest within this (frames).
PAIR_FRAMES = 2.0
#: Witness: detector bounce-class probability on a row round(f)-1..round(f)+1 at least this.
WITNESS_P = 0.2
#: A state is past the validated end when its frame exceeds ``end_frame`` by more than this.
CONTINUATION_EPS = 1e-3
CLASSES = ("none", "contact", "bounce", "net_hit")
PHYSICAL = ("contact", "bounce", "net_hit")
NET_HIT = "fit_only_bounce_at_net_hit"
UNWITNESSED = "fit_only_bounce_unwitnessed"
REASONS = (NET_HIT, UNWITNESSED)
CONTINUATION = "modelled_continuation"


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    value = (os.environ if environ is None else environ).get(SWITCH, "off")
    if value not in ("on", "off"):
        raise ValueError(f"{SWITCH} must be on or off, not {value!r}")
    return value == "on"


def to_rules_json() -> dict:
    """The frozen definition as data (``RULES.json``)."""
    return {
        "rules_version": RULES_VERSION,
        "switch": SWITCH,
        "constants": {
            "join_frames": JOIN_FRAMES,
            "pair_frames": PAIR_FRAMES,
            "witness_p": WITNESS_P,
            "continuation_eps": CONTINUATION_EPS,
        },
        "reasons": list(REASONS),
        "continuation_label": CONTINUATION,
        "definition": __doc__,
    }


# ---------------------------------------------------------------- detector record


def record_frame(record: dict) -> float:
    sub = (record.get("location") or {}).get("frame_subpixel")
    return float(sub if sub is not None else record["frame"])


def _clip_key(clip) -> str:
    return str(clip).rsplit("__", 1)[-1]


class DetectorRecord:
    """The event detector's emission records and per-row class probabilities for one match."""

    def __init__(self, emissions: list[dict], predictions: Mapping):
        self.by_clip: dict[str, list[dict]] = defaultdict(list)
        for r in emissions:
            if r.get("event_type") in PHYSICAL:
                self.by_clip[_clip_key(r["clip"])].append(r)
        self.rows: dict[str, dict[int, int]] = defaultdict(dict)
        for i, (c, f) in enumerate(zip(predictions["clips"], predictions["frames"], strict=True)):
            self.rows[_clip_key(c)][int(f)] = i
        self.prob = np.asarray(predictions["probabilities"])

    @classmethod
    def load(cls, upstream: Path) -> DetectorRecord:
        """From a match's ``upstream/event_emissions.json`` and ``.predictions.npz``."""
        emissions = json.loads((upstream / "event_emissions.json").read_text())
        return cls.from_bytes(
            emissions, (upstream / "event_emissions.predictions.npz").read_bytes()
        )

    @classmethod
    def from_bytes(cls, emissions: list[dict], predictions_npz: bytes) -> DetectorRecord:
        with np.load(io.BytesIO(predictions_npz), allow_pickle=False) as z:
            predictions = {k: z[k] for k in ("clips", "frames", "probabilities")}
        return cls(emissions, predictions)

    def near(self, clip: str, frame: float) -> list[dict]:
        """Physical records within ``JOIN_FRAMES`` of ``frame``."""
        return [
            r
            for r in self.by_clip.get(_clip_key(clip), [])
            if abs(record_frame(r) - frame) <= JOIN_FRAMES
        ]

    def bounce_p_pm1(self, clip: str, frame: float) -> float | None:
        rows = self.rows.get(_clip_key(clip), {})
        centre = int(round(frame))
        k = CLASSES.index("bounce")
        near = [float(self.prob[rows[f], k]) for f in (centre - 1, centre, centre + 1) if f in rows]
        return round(max(near), 6) if near else None


def detector_status(record: dict) -> str:
    if not record.get("abstain"):
        return "emitted"
    return "model_abstain" if record.get("model_abstain") else "gate_held"


# ---------------------------------------------------------------- classification


def classify_bounce(
    fitted_frame: float,
    end_frame: float,
    *,
    supplied_frame: float | None,
    packet_bounce_near: bool,
    records_near: list[dict],
    bounce_p_pm1: float | None,
) -> dict:
    """One modelled bounce of a validated flight: position, fit-only, witness, rule reason.

    ``records_near``: the detector's physical records within ``JOIN_FRAMES`` (any type/status).
    """
    position = "inside" if fitted_frame < end_frame else "after"
    fit_only = supplied_frame is None and not packet_bounce_near
    others = [r for r in records_near if r.get("event_type") != "bounce"]
    emitted_net = [
        r for r in others if r.get("event_type") == "net_hit" and detector_status(r) == "emitted"
    ]
    witnessed = not others and (bounce_p_pm1 or 0.0) >= WITNESS_P
    reason = None
    if fit_only and position == "inside":
        if emitted_net:
            reason = NET_HIT
        elif not witnessed:
            reason = UNWITNESSED
    return {
        "fitted_frame": round(float(fitted_frame), 4),
        "supplied_frame": None if supplied_frame is None else round(float(supplied_frame), 4),
        "position": position,
        "fit_only": fit_only,
        "witnessed": witnessed,
        "bounce_p_pm1": bounce_p_pm1,
        "other_type_records": [
            {
                "event_type": r["event_type"],
                "frame": round(record_frame(r), 4),
                "status": detector_status(r),
            }
            for r in sorted(others, key=lambda r: abs(record_frame(r) - fitted_frame))
        ],
        "detector_bounce_near": any(r.get("event_type") == "bounce" for r in records_near),
        "reason": reason,
    }


def pair(supplied: list[float], modelled: list[float]) -> list[tuple[float | None, float | None]]:
    """Pair supplied and modelled epochs in order (equal counts are the normal case)."""
    if len(supplied) == len(modelled):
        return list(zip(supplied, modelled, strict=True))
    out, left = [], list(modelled)
    for s in supplied:
        if left:
            m = min(left, key=lambda v: abs(v - s))
            if abs(m - s) <= PAIR_FRAMES:
                left.remove(m)
                out.append((s, m))
                continue
        out.append((s, None))
    out += [(None, m) for m in left]
    return out


def span_key(flight: Mapping) -> tuple[float, float]:
    return (round(float(flight["start_frame"]), 4), round(float(flight["end_frame"]), 4))


def fit_cases(
    result: dict, packet_events: list[dict] | None
) -> list[tuple[dict, list[int], list[dict]]]:
    """(per-case result with ``verdict``+``measurement``, local->original flight map, events)."""
    out = []
    if result.get("schema") == "labeled_s6_contact_component_source_v1":
        events = result["source_plan"]["original_events"]
        for cr in result.get("component_results") or []:
            res = cr.get("result") or {}
            if "verdict" in res and "measurement" in res:
                out.append((res, list(cr["local_to_original"]["flights"]), events))
    else:
        events = packet_events or []
        n = len((result.get("verdict") or {}).get("flights") or [])
        if "measurement" in result:
            out.append((result, list(range(n)), events))
    merged = {
        span_key(f): i for i, f in enumerate((result.get("verdict") or {}).get("flights") or [])
    }
    for record in (result.get("split_fallback") or {}).get("records") or []:
        res = record.get("result") or {}
        if "verdict" in res and "measurement" in res:
            spans = [span_key(f) for f in res["verdict"]["flights"]]
            out.append((res, [merged.get(sp, -1) for sp in spans], events))
    return out


def packet_events(packet: dict) -> list[dict]:
    """Fit-packet events (the fields the rules read) of a first-fit ``inputs/packet.json``."""
    return [
        {k: e.get(k) for k in ("event_type", "frame", "note", "status", "annotation_origin")}
        for att in packet.get("attempts", [])
        for e in att.get("events", [])
    ]


def _packet_bounce_near(events: list[dict], frame: float) -> bool:
    return any(
        e.get("event_type") == "bounce" and abs(float(e["frame"]) - frame) <= JOIN_FRAMES
        for e in events
    )


def flight_bounces(case: dict, local: int, events: list[dict]) -> list[tuple[float | None, float]]:
    """(supplied, fitted) bounce epochs of one fitted flight; supplied-only ones are dropped
    (the fit carried them but modelled no ground contact)."""
    vf = case["verdict"]["flights"][local]
    dense = case["measurement"]["dense_flights"][local]
    if abs(float(dense["start_frame"]) - float(vf["start_frame"])) > 1e-6:
        raise ValueError("dense flight and verdict flight disagree")
    pairs = pair(
        [
            float(b["supplied_frame"])
            for b in vf.get("bounce_witness") or []
            if "supplied_frame" in b
        ],
        [float(b["frame"]) for b in dense.get("bounces") or []],
    )
    return [(s, m) for s, m in pairs if m is not None]


def evaluate_result(
    result: dict,
    packet_events: list[dict] | None,
    detector: DetectorRecord,
    clip: str,
    validated: list[tuple[float, float]],
) -> dict[tuple[float, float], dict]:
    """Rule decision per validated flight span (``span_key`` of the merged verdict flight).

    ``{"demote": bool, "reason": first reason or None, "reasons": [...], "bounces": [...],
    "verdict_end_frame": float}``; spans with no accepting fit case are omitted.
    """
    wanted = set(validated)
    chosen: dict[tuple, tuple] = {}
    for case, mapping, events in fit_cases(result, packet_events):
        for local, _ in enumerate(mapping):
            vf = case["verdict"]["flights"][local]
            span = span_key(vf)
            if span in wanted and vf.get("accepted") and span not in chosen:
                chosen[span] = (case, local, events)
    out = {}
    for span, (case, local, events) in sorted(chosen.items()):
        vf = case["verdict"]["flights"][local]
        end = float(vf["end_frame"])
        bounces = []
        for supplied, fitted in flight_bounces(case, local, events):
            ref = supplied if supplied is not None else fitted
            bounces.append(
                classify_bounce(
                    fitted,
                    end,
                    supplied_frame=supplied,
                    packet_bounce_near=_packet_bounce_near(events, ref),
                    records_near=detector.near(clip, fitted),
                    bounce_p_pm1=detector.bounce_p_pm1(clip, fitted),
                )
            )
        reasons = sorted({b["reason"] for b in bounces if b["reason"]}, key=REASONS.index)
        out[span] = {
            "rules_version": RULES_VERSION,
            "demote": bool(reasons),
            "reason": reasons[0] if reasons else None,
            "reasons": reasons,
            "verdict_end_frame": end,
            "bounces": bounces,
        }
    return out
