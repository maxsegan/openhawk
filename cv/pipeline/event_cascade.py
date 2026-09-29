"""Paid event cascade, FREEZE v2: which rows are asked, and what the answers decide.

After the automatic S6 fit, a decoder-emitted event the planner would not use
(a retained abstention or another unsupported occurrence) that bounds or sits
inside a flight the fit did not accept is a candidate. Each candidate climbs
local Qwen (two draws), Gemini Flash (two draws), then Opus (one draw). The
first tier whose draws all confirm the emitted type within two frames, or all
reject, settles it. A confirm makes the row a supported claim, a reject removes
it, and an unsettled row is left as it was. FREEZE v2: a reject that
contradicts the ball track climbs instead, and after Opus it stays unsettled.

Which flights count as not accepted is ``candidate_scope`` (``CANDIDATE_SCOPES``):
``unresolved`` (the recorded FREEZE v2 rule) reads only the plan's slot status, so a
slot an ending or ground default prepared but the fit rejected is never asked. The
wider scopes also read the fit verdict. The runner default is
``DEFAULT_CANDIDATE_SCOPE`` (``unresolved``).

Two cost switches, both measured on panel D (FINAL_REPORT_cascade_cost.md):

``second_draw``  ``settle_only`` (default) asks a tier's second draw only when
                 the first could still settle; decisions are identical by
                 construction. ``always`` is the recorded FREEZE v2 behaviour.
``opus_evidence`` ``1280`` (default) shows Opus both evidence grids resized to
                 1280 px. ``full`` sends the 6400 px grids.

``first_tier``   ``gemini`` (default since 2026-09-28, owner decision) drops the
                 local Qwen tier: Flash first, then Opus, same prompts, evidence
                 and settle rule (cascade-no-qwen lane: pooled +11 vs +15 flights,
                 ~$3.70/match, no Qwen GPU). ``qwen`` is the recorded FREEZE v2.

``adjudication``   who settles a candidate. ``paid`` walks the model tiers above.
                 ``fitter`` asks no model: every candidate is confirmed with
                 provenance ``fitter_adjudicated`` ($0) and the refit's own
                 physics gates accept or reject the flights it bounds. Its
                 candidates use ``FITTER_CANDIDATE_SCOPE`` (the narrow paid scope
                 was a cost choice). Measured in the aa80 plan (probe 1) and
                 FINAL_REPORT_fitter_adjudication.md.

Nothing here reads an evaluation label. Candidates come from the fit's own
component plan and the automatic emissions. Ported from the cascade-rerun
branch (``event_cascade.adjudicate``, ``fresh_candidates``, ``unsupported``),
``held_release/contradictory.py`` and ``cascade_cost/replay.py``.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from cv.pipeline import resolution
from cv.pipeline import s6_first_contact_role as role

SCHEMA = "event_cascade_freeze_v2"
DECISIONS_SCHEMA = "event_cascade_adjudication_v1"
CANDIDATES_SCHEMA = "event_cascade_candidates_v1"
PHYSICAL = ("contact", "bounce", "net_hit")
REJECT_TYPES = ("none", "no_event")
FRAME_WINDOW = 2.0
BAND_LOW = 0.78
FRAMES = "audit_frames_native_1080"
TRACK = "ball_track_joint_native1080_arc_augmented_v2.csv"

SECOND_DRAW_CHOICES = ("settle_only", "always")
#: Which flights' rows are asked. ``unresolved`` (recorded FREEZE v2): slots the
#: plan did not prepare. ``unresolved_or_rejected`` also asks on prepared slots the
#: fit did not accept (no accepted, emitted flight). ``unresolved_or_uncovered``
#: further asks on the part of a prepared slot no accepted flight covers (a flight
#: closed early at an ambiguous row).
CANDIDATE_SCOPES = ("unresolved", "unresolved_or_rejected", "unresolved_or_uncovered")
#: cascade-coverage (FINAL_REPORT_cascade_coverage.md) measured ``unresolved_or_rejected`` at
#: +4 flights over fresh/C/D/E with 0 new wrong accepts, but +$0.2-1.1 per match (D cascade
#: $1.74 -> $2.82 of the ~$3.33 budget). Kept off for cost; opt in with --candidate-scope.
DEFAULT_CANDIDATE_SCOPE = "unresolved"
OPUS_EVIDENCE_CHOICES = ("1280", "full")
FIRST_TIER_CHOICES = ("qwen", "gemini")
DEFAULT_SETTINGS = {"second_draw": "settle_only", "opus_evidence": "1280", "first_tier": "gemini"}
#: ``fitter``: every candidate confirmed without a model call; the refit decides.
#: ``paid``: the model tiers (FREEZE v2 walk). Rollback: ``paid``.
#: Default ``fitter`` since 2026-09-28 (FINAL_REPORT_fitter_adjudication.md: 465 vs 449
#: flights of 679 against Flash-first on five development panels, same unmatched accepts).
ADJUDICATION_CHOICES = ("fitter", "paid")
DEFAULT_ADJUDICATION = "fitter"
#: Source name a fitter-adjudicated confirm carries in place of a model tier.
FITTER_SOURCE = "fitter_adjudicated"
#: Candidate scope of fitter adjudication (aa80 probe 1).
FITTER_CANDIDATE_SCOPE = "unresolved_or_rejected"
#: The recorded combined-v2 policy. Rollback: pass these settings.
RECORDED_SETTINGS = {"second_draw": "always", "opus_evidence": "full"}

# FREEZE v2 contradictory reject (held_release/contradictory.py).
CONTACT_MARGINAL = 0.95
CONTACT_LEG_PX = 8.0
BOUNCE_VERTICAL_PX = 6.0
TRACK_LEG_FRAMES = 4
TRACK_SPACE = resolution.LEGACY_TRACKING_SIZE


@dataclass(frozen=True)
class Tier:
    name: str
    number: int
    model: str
    profile: str
    draws: tuple[int, ...]
    max_tokens: tuple[int, ...]
    reasoning: str | None


TIERS = (
    Tier("qwen", 1, "Qwen/Qwen3.8-27B-FP8", "guided_nothink", (1, 2), (2048, 4096), None),
    Tier("gemini", 2, "google/gemini-3.8-flash", "zero", (1, 2), (8192, 16384), "low"),
    Tier("opus", 3, "anthropic/claude-opus-5.5", "zero", (1,), (8192, 16384), "low"),
)
TIER_BY_NAME = {tier.name: tier for tier in TIERS}


def settings(overrides: dict | None = None) -> dict:
    """The cascade switches. Unknown keys or values raise."""
    result = dict(DEFAULT_SETTINGS)
    for key, value in (overrides or {}).items():
        if key not in result:
            raise ValueError(f"unknown cascade setting {key}")
        result[key] = str(value)
    if result["second_draw"] not in SECOND_DRAW_CHOICES:
        raise ValueError("second_draw must be settle_only or always")
    if result["opus_evidence"] not in OPUS_EVIDENCE_CHOICES:
        raise ValueError("opus_evidence must be 1280 or full")
    if result["first_tier"] not in FIRST_TIER_CHOICES:
        raise ValueError("first_tier must be qwen or gemini")
    return result


def active_tiers(chosen: dict) -> tuple[Tier, ...]:
    """The tiers a walk climbs: all three, or Flash then Opus with ``first_tier=gemini``."""
    names = [tier.name for tier in TIERS]
    return TIERS[names.index(chosen["first_tier"]) :]


def opus_side(chosen: dict) -> int | None:
    return None if chosen["opus_evidence"] == "full" else int(chosen["opus_evidence"])


def reply_profile(tier: str, chosen: dict) -> str:
    """Reply directory name. A resized Opus reply never shares the full-size cache."""
    profile = TIER_BY_NAME[tier].profile
    side = opus_side(chosen) if tier == "opus" else None
    return profile if side is None else f"{profile}@{side}"


# --- answers ---------------------------------------------------------------


@dataclass(frozen=True)
class Answer:
    event_type: str | None
    frame: float | None
    usd: float = 0.0
    tokens: dict | None = None
    # Recorded, never read. Confidence is not a trigger.
    confidence: str | None = None


def answer_from_reply(doc: dict) -> Answer:
    """One saved labeller reply."""
    verdict = doc.get("verdict") or {}
    kind = verdict.get("event_type")
    if not isinstance(kind, str):
        kind = None
    raw_frame = verdict.get("frame")
    try:
        frame = None if raw_frame is None else float(raw_frame)
    except (TypeError, ValueError):
        frame = None
    usage = doc.get("usage") or {}
    confidence = verdict.get("confidence")
    return Answer(
        kind,
        frame,
        usd=float(doc.get("usd") or 0),
        tokens={"prompt": usage.get("prompt_tokens"), "completion": usage.get("completion_tokens")},
        confidence=confidence if isinstance(confidence, str) else None,
    )


def reply_is_good(path: Path) -> bool:
    """The frozen reader (run_label.already_good): a truncated reply is no answer."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    doc = json.loads(path.read_text())
    if doc.get("finish_reason") in {"length", "max_tokens"}:
        return False
    return isinstance(doc.get("verdict"), dict) and "event_type" in doc["verdict"]


def read_answer(path: Path) -> Answer | None:
    if not reply_is_good(path):
        return None
    return answer_from_reply(json.loads(path.read_text()))


def _is_emit(answer: Answer) -> bool:
    return answer.event_type in PHYSICAL and answer.frame is not None


def _is_reject(answer: Answer) -> bool:
    return answer.event_type in REJECT_TYPES


def _is_settled(answer: Answer) -> bool:
    return _is_emit(answer) or _is_reject(answer)


def agrees(left: Answer, right: Answer, window: float = FRAME_WINDOW) -> bool:
    if not _is_settled(left) or not _is_settled(right):
        return False
    if _is_reject(left) and _is_reject(right):
        return True
    if _is_emit(left) and _is_emit(right) and left.event_type == right.event_type:
        return abs(float(left.frame) - float(right.frame)) <= window
    return False


def occurrence_decision(
    answers: list[Answer], *, emitted_type: str, emitted_frame: float, window: float = FRAME_WINDOW
) -> str:
    """``confirm``, ``reject`` or ``unsettled`` for one tier's draws."""
    if not answers:
        raise ValueError("answers")
    if emitted_type not in PHYSICAL:
        raise ValueError("emitted type")
    if any(not _is_settled(answer) for answer in answers):
        return "unsettled"
    if any(not agrees(answers[0], answer, window) for answer in answers[1:]):
        return "unsettled"
    if _is_reject(answers[0]):
        return "reject"
    if answers[0].event_type != emitted_type:
        return "unsettled"
    if abs(float(answers[0].frame) - float(emitted_frame)) > window:
        return "unsettled"
    return "confirm"


def draw_one_can_settle(answer: Answer, row: dict) -> bool:
    """False when draw 1 alone already makes the tier unsettled (saving A)."""
    return (
        occurrence_decision(
            [answer], emitted_type=row["event_type"], emitted_frame=float(row["frame"])
        )
        != "unsettled"
    )


def _climb_reason(answers: list[Answer], row: dict, action: str) -> str:
    if action == "reject":
        return "contradictory_reject"
    if any(not _is_settled(answer) for answer in answers):
        return "unsettled"
    if len(answers) >= 2 and not all(agrees(answers[0], answer) for answer in answers[1:]):
        return "draw_disagreement"
    return "tier_disagreement"


def receipt(tier: Tier, answer: Answer, *, trigger: str, candidate_id: str, draw: int) -> dict:
    return {
        "tier": tier.number,
        "model": tier.model,
        "tokens": dict(answer.tokens or {}),
        "usd": float(answer.usd),
        "answer": {"event_type": answer.event_type, "frame": answer.frame},
        "trigger": trigger,
        "candidate_id": str(candidate_id),
        "draw": int(draw),
    }


def settle(row: dict, answer_for, *, contradicts: bool, chosen: dict | None = None) -> dict:
    """Walk the tiers for one candidate.

    ``answer_for(tier_name, draw)`` returns a saved ``Answer`` or None. None
    means the call is still owed: the result then has ``owed`` set and no
    action. Receipts cover every draw the walk used.
    """
    chosen = settings(chosen)
    emitted_type = row["event_type"]
    emitted_frame = float(row["frame"])
    receipts: list[dict] = []
    trigger = "enter"
    climbed_reject = False
    for tier in active_tiers(chosen):
        answers: list[Answer] = []
        for index, draw in enumerate(tier.draws):
            answer = answer_for(tier.name, draw)
            if answer is None:
                return {"owed": (tier.name, draw), "receipts": receipts}
            answers.append(answer)
            receipts.append(
                receipt(
                    tier,
                    answer,
                    trigger=trigger if index == 0 else "second_draw",
                    candidate_id=row["id"],
                    draw=draw,
                )
            )
            if (
                chosen["second_draw"] == "settle_only"
                and index == 0
                and len(tier.draws) > 1
                and not draw_one_can_settle(answer, row)
            ):
                break
        action = occurrence_decision(
            answers, emitted_type=emitted_type, emitted_frame=emitted_frame
        )
        if action == "confirm" or (action == "reject" and not contradicts):
            return {
                "owed": None,
                "action": action,
                "tier": tier.name,
                "model": tier.model,
                "receipts": receipts,
            }
        climbed_reject = climbed_reject or action == "reject"
        trigger = _climb_reason(answers, row, action)
    # After Opus the row stays unsupported. A walk that climbed past a
    # contradictory reject names the last tier, as the recorded FREEZE v2 did.
    return {
        "owed": None,
        "action": "unsettled",
        "tier": TIERS[-1].name if climbed_reject else None,
        "model": None,
        "receipts": receipts,
    }


def candidate_scope_for(adjudication: str, scope: str | None = None) -> str:
    """The candidate scope: explicit, else the adjudication's default."""
    if adjudication not in ADJUDICATION_CHOICES:
        raise ValueError(f"adjudication must be one of {ADJUDICATION_CHOICES}")
    if scope is not None:
        return scope
    return FITTER_CANDIDATE_SCOPE if adjudication == "fitter" else DEFAULT_CANDIDATE_SCOPE


def fitter_settle(row: dict) -> dict:
    """Fitter adjudication of one candidate: confirmed, no model asked, no receipt."""
    return {
        "owed": None,
        "action": "confirm",
        "tier": FITTER_SOURCE,
        "model": None,
        "receipts": [],
    }


# --- FREEZE v2 track contradiction ------------------------------------------


def contradictory_reject(*, event_type: str, marginal: float | None, pre, post) -> bool:
    """True when local track evidence contradicts a tier reject."""

    def leg(velocity):
        if velocity is None or len(velocity) != 2:
            return None
        x, y = float(velocity[0]), float(velocity[1])
        return (x, y) if math.isfinite(x) and math.isfinite(y) else None

    before, after = leg(pre), leg(post)
    if before is None or after is None:
        return False
    if event_type == "contact":
        if marginal is None or marginal < CONTACT_MARGINAL:
            return False
        dot = before[0] * after[0] + before[1] * after[1]
        return (
            math.hypot(*before) >= CONTACT_LEG_PX
            and math.hypot(*after) >= CONTACT_LEG_PX
            and dot < 0
        )
    if event_type == "bounce":
        if before[1] * after[1] >= 0:
            return False
        if abs(before[1]) < BOUNCE_VERTICAL_PX or abs(after[1]) < BOUNCE_VERTICAL_PX:
            return False
        # A lateral reversal is a hit, not a bounce.
        if before[0] * after[0] < 0 and abs(before[0]) >= 4 and abs(after[0]) >= 4:
            return False
        return True
    return False


def track_points(path: Path, clip: str) -> dict[int, tuple[float, float]]:
    """Observed (not interpolated) track samples of one point clip.

    The FREEZE v2 leg thresholds were set on the track's legacy ``x``/``y``
    columns, which the sidecar declares as 960x540. Refuse any other space
    rather than rescale the rows and leave the thresholds behind.
    """
    resolution.require_artifact_space(
        path, TRACK_SPACE, columns=("x", "y"), consumer="event cascade track contradiction"
    )
    points = {}
    with path.open() as handle:
        for sample in csv.DictReader(handle):
            name = sample.get("clip") or ""
            if name != clip and not name.endswith(clip):
                continue
            stem = Path(sample["frame"]).stem
            frame = int(stem[2:]) if stem.startswith("f_") else int(float(stem))
            if "interpolated" in (sample.get("sources") or ""):
                continue
            points[frame] = (float(sample["x"]), float(sample["y"]))
    return points


def _leg(points: dict[int, tuple[float, float]], start: int, end: int):
    if start not in points or end not in points or start == end:
        return None
    x0, y0 = points[start]
    x1, y1 = points[end]
    return ((x1 - x0) / (end - start), (y1 - y0) / (end - start))


def contradicts_track(row: dict, points: dict[int, tuple[float, float]]) -> bool:
    frame = int(round(float(row["frame"])))
    return contradictory_reject(
        event_type=row["event_type"],
        marginal=row.get("marginal"),
        pre=_leg(points, frame - TRACK_LEG_FRAMES, frame),
        post=_leg(points, frame, frame + TRACK_LEG_FRAMES),
    )


# --- candidates -------------------------------------------------------------


def marginal_of(event: dict) -> float | None:
    recorded = (event.get("automatic_abstention") or {}).get("acceptance_marginal")
    if isinstance(recorded, bool) or not isinstance(recorded, (int, float)):
        return None
    return float(recorded)


def band_of(marginal: float | None, *, gate_held: bool = False, low: float = BAND_LOW) -> str:
    if gate_held:
        return "gate_held"
    if marginal is None:
        return "no_marginal"
    return "marginal_band" if marginal >= low else "below_band"


def why_unsupported(event: dict) -> str | None:
    """Why the planner will not use this row, or None when it already will."""
    if role.supported(event):
        return None
    status = event.get("occurrence_status")
    if status == "ambiguous" and event.get("automatic_abstention"):
        return "retained_model_abstention"
    if status in ("uncertain", "ambiguous", "absent", "unsupported"):
        return f"occurrence_status={status}"
    return "membership_refused"


def placements(frame: float, slots: list[dict]) -> list[str]:
    places = []
    for slot in slots:
        start = float(slot["start_frame"])
        end = float(slot["end_frame"])
        if abs(frame - start) <= 1e-3:
            places.append(f"origin:{slot['original_contact_index']}")
        elif abs(frame - end) <= 1e-3 and slot.get("supplied_next_contact_index") is not None:
            places.append(f"endpoint:{slot['original_contact_index']}")
        elif start < frame < end:
            places.append(f"inside:{slot['original_contact_index']}")
    return places


def emission_index(path: Path) -> dict[tuple, dict]:
    """Automatic decoder emissions keyed by clip, type and rounded frame."""
    document = json.loads(path.read_text())
    if isinstance(document, dict):
        document = document["emissions"]
    index = {}
    for row in document:
        if row.get("event_type") not in PHYSICAL:
            continue
        try:
            frame = float(row["frame"])
        except (KeyError, TypeError, ValueError):
            continue
        index[(row.get("clip"), row.get("event_type"), frame)] = row
    return index


def emission_row(event: dict, plan: dict, index: dict[tuple, dict]) -> dict | None:
    frame = event.get("emitted_rounded_frame")
    try:
        rounded = float(frame) if frame is not None else float(round(float(event["frame"])))
    except (TypeError, ValueError):
        return None
    for clip in (f"{plan['match_id']}__{plan['clip']}", plan["clip"]):
        row = index.get((clip, event.get("event_type"), rounded))
        if row is not None:
            return row
    return None


def accepted_spans(verdict: dict) -> dict[str, list[tuple[float, float]]]:
    """Accepted, emitted flights of a fit verdict, keyed by original contact index."""
    spans: dict[str, list[tuple[float, float]]] = {}
    for flight in verdict.get("flights") or []:
        if not flight.get("accepted") or flight.get("emitted") is False:
            continue
        if flight.get("original_contact_index") is None:
            continue
        spans.setdefault(str(flight["original_contact_index"]), []).append(
            (float(flight["start_frame"]), float(flight["end_frame"]))
        )
    return spans


def lost_places(
    frame: float, slots: list[dict], *, scope: str = "unresolved", verdict: dict | None = None
) -> list[str]:
    """The placements of ``frame`` on flights the fit did not accept, under ``scope``."""
    if scope not in CANDIDATE_SCOPES:
        raise ValueError(f"candidate scope must be one of {CANDIDATE_SCOPES}")
    if scope != "unresolved" and verdict is None:
        raise ValueError(f"candidate scope {scope} needs the fit verdict")
    unprepared = {
        str(slot["original_contact_index"]) for slot in slots if slot.get("status") != "prepared"
    }
    accepted = {} if verdict is None else accepted_spans(verdict)
    found = []
    for place in placements(frame, slots):
        slot = place.split(":")[1]
        if slot in unprepared:
            found.append(place)
        elif scope == "unresolved":
            continue
        elif slot not in accepted:
            found.append(place)
        elif scope == "unresolved_or_uncovered" and not any(
            start - 1e-3 <= frame <= end + 1e-3 for start, end in accepted[slot]
        ):
            found.append(place)
    return found


def select_from_plan(
    plan: dict,
    emissions: dict | None = None,
    *,
    scope: str = "unresolved",
    verdict: dict | None = None,
) -> list[dict]:
    """Unsupported rows on flights the plan (``unresolved``) or the fit did not accept.

    ``emissions`` only names the band (gate-held or not). It cannot add or drop a row.
    ``verdict`` is the fit's ``result.json`` verdict, needed by the wider scopes.
    """
    found = []
    for index, event in enumerate(plan.get("original_events") or []):
        reason = why_unsupported(event)
        if reason is None:
            continue
        on_lost = lost_places(
            float(event["frame"]), plan["original_slots"], scope=scope, verdict=verdict
        )
        if not on_lost:
            continue
        source = None if emissions is None else emission_row(event, plan, emissions)
        gate_held = bool(source and source.get("gate_held") is True)
        found.append(
            {
                "event_index": index,
                "event_type": event["event_type"],
                "frame": float(event["frame"]),
                "why": reason,
                "places": on_lost,
                "marginal": marginal_of(event),
                "gate_held": gate_held,
                "band": band_of(marginal_of(event), gate_held=gate_held),
                "emission_found": source is not None,
            }
        )
    return found


def match_directory(pose: Path) -> Path:
    """The match directory holding native frames and the ball track for a pose file."""
    if (pose.parent / FRAMES).is_dir() and (pose.parent / TRACK).is_file():
        return pose.parent
    sidecar = Path(str(pose) + ".coordinates.json")
    if not sidecar.is_file():
        raise FileNotFoundError(
            f"match directory is not {pose.parent} and {pose.name} has no sidecar"
        )
    source = json.loads(sidecar.read_text()).get("source")
    if not source:
        raise FileNotFoundError(f"{sidecar} does not name the boxes it was copied from")
    match = Path(source).parent
    if (match / FRAMES).is_dir() and (match / TRACK).is_file():
        return match
    raise FileNotFoundError(f"no native frames and ball track for {pose} or {match}")


def candidate_rows(
    job: dict,
    plan: dict,
    match: Path,
    *,
    scope: str = "unresolved",
    verdict: dict | None = None,
) -> list[dict]:
    """Candidate records for one fitted attempt."""
    emissions = emission_index(match.parent / "event_emissions.json")
    rows = []
    for decision in select_from_plan(plan, emissions, scope=scope, verdict=verdict):
        event = plan["original_events"][decision["event_index"]]
        rounded = event.get("emitted_rounded_frame", round(float(event["frame"])))
        point = plan["clip"]
        rows.append(
            {
                "id": f"{job['attempt']}__{decision['event_index']}",
                "attempt": job["attempt"],
                "panel": job["panel"],
                "match_id": plan["match_id"],
                "point": point,
                "clip": f"{plan['match_id']}__{point}",
                "event_index": decision["event_index"],
                "event_type": decision["event_type"],
                "frame": decision["frame"],
                "nominated_frame": int(round(float(rounded))),
                "band": decision["band"],
                "marginal": decision["marginal"],
                "gate_held": decision["gate_held"],
                "why": decision["why"],
                "places": decision["places"],
                "frame_dir": str(match / FRAMES / point),
                "track_csv": str(match / TRACK),
                "layout": "one",
            }
        )
    return rows
