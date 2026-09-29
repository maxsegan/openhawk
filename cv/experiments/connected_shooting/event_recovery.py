"""Label-free event hypotheses from connected-fit residual structure.

Question this answers: when the supplied event topology is incomplete, noisy or
wrong, can the connected search say *where* an event is missing, *which kinds*
are consistent there, and *which supplied event* its own physics cannot explain?

Everything here is input-only.  Proposals read the training image residuals of a
fit that already ran, the same native fronts the objective used, and the point
grammar.  Withheld pictures, evaluation labels and the removed-event identity of
a robustness variant are never consulted, so recovery precision and recall
cannot be inflated by the answer key.

Nothing here selects silently.  ``topology_hypotheses`` returns every branch it
would try, each with the reason it exists, and the caller reports the survivors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from cv.experiments.connected_shooting import player_position
from cv.experiments.connected_shooting import real_bidirectional_search as whole


KINDS = ("contact", "bounce", "net_hit", "ending")
NET_PLANE_Y_M = 11.885
NET_HALF_WIDTH_M = 6.4
NET_CENTRE_X_M = 5.485

# Declared proposal controls.  They are explicit research settings, not
# calibrated detector thresholds; every one of them appears in the receipt.
RESIDUAL_FLOOR_PX = 6.0
RESIDUAL_MULTIPLE_OF_MEDIAN = 3.0
MINIMUM_RUN_PICTURES = 2
GROUND_RAY_HEIGHT_M = 0.70
NET_PLANE_TOLERANCE_M = 0.60
CONTACT_PLAYER_RADIUS_M = 2.5

# Grammar priors, as negative log weights added to the input-only rank score.
# Larger means "less expected under ordinary singles play", never "forbidden".
GRAMMAR_PENALTIES = {
    "bounce_between_contacts": 0.0,
    "volley_contact_without_preceding_bounce": 1.0,
    "contact_breaks_alternating_hitters": 3.0,
    "second_bounce_before_the_ending": 0.0,
    "net_hit_inside_one_flight": 0.5,
    "ending_at_the_last_supported_picture": 0.0,
    "demote_a_supplied_event": 2.0,
    "add_a_recovered_event": 2.0,
    # A point that ends in the net still ends with the ball on the ground.  The
    # three charges below rank the readings of that ending against each other;
    # none of them is free, and the truncation that drops a labeled contact is
    # deliberately the most expensive branch the beam will ever take.
    "extend_to_a_labeled_dead_ball_bounce": 1.0,
    "terminal_ground_impact_at_the_ending": 1.5,
    "truncate_to_the_last_mid_rally_bounce": 4.0,
}

# The image witness for a ground impact at an ambiguous terminal epoch: the
# front's native row must reverse its vertical direction inside this bracket.
TERMINAL_REVERSAL_BRACKET_FRAMES = 3


@dataclass(frozen=True)
class Proposal:
    """One unexplained span with its consistent kinds and timing range."""

    frame: float
    frame_interval: tuple[float, float]
    kind_likelihoods: dict[str, float]
    evidence: dict
    source: str = "training_residual_run"

    def kinds(self, minimum: float = 0.05) -> list[str]:
        ordered = sorted(self.kind_likelihoods.items(), key=lambda row: (-row[1], row[0]))
        return [kind for kind, weight in ordered if weight >= minimum]

    def as_dict(self) -> dict:
        return {
            "frame": self.frame,
            "frame_interval": list(self.frame_interval),
            "kind_likelihoods": self.kind_likelihoods,
            "consistent_kinds": self.kinds(),
            "evidence": self.evidence,
            "source": self.source,
        }


@dataclass
class Hypothesis:
    """One complete candidate topology the search will actually fit."""

    name: str
    events: list[dict]
    added: list[dict] = field(default_factory=list)
    demoted: list[dict] = field(default_factory=list)
    grammar_penalty: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "event_count": len(self.events),
            "added_events": self.added,
            "demoted_events": self.demoted,
            "grammar_penalty": self.grammar_penalty,
            "reasons": self.reasons,
        }


def residual_runs(
    native_projection: list[dict],
    *,
    floor_px: float = RESIDUAL_FLOOR_PX,
    multiple: float = RESIDUAL_MULTIPLE_OF_MEDIAN,
    minimum_pictures: int = MINIMUM_RUN_PICTURES,
) -> list[dict]:
    """Contiguous training pictures the connected fit does not explain.

    Only the training split is read.  A withheld picture must never influence a
    branch the selector then ranks, so it cannot influence a proposal either.
    """
    rows = sorted(
        (row for row in native_projection if row["split"] == "training"),
        key=lambda row: float(row["frame"]),
    )
    if len(rows) < minimum_pictures + 1:
        return []
    errors = np.asarray([float(row["error_px"]) for row in rows])
    threshold = max(float(floor_px), multiple * float(np.median(errors)))
    flagged = errors > threshold
    runs = []
    start = None
    for index, value in enumerate(flagged):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(flagged) - 1))
    result = []
    for low, high in runs:
        if high - low + 1 < minimum_pictures:
            continue
        before = float(rows[low - 1]["frame"]) if low else float(rows[low]["frame"]) - 1.0
        result.append(
            {
                "onset_bracket_frames": [before, float(rows[low]["frame"])],
                "start_frame": float(rows[low]["frame"]),
                "end_frame": float(rows[high]["frame"]),
                "pictures": high - low + 1,
                "maximum_error_px": float(errors[low : high + 1].max()),
                "median_error_px": float(np.median(errors[low : high + 1])),
                "threshold_px": threshold,
                "reaches_last_training_picture": high == len(rows) - 1,
            }
        )
    return result


def ground_ray_state(
    frame: float, cameras: dict[int, np.ndarray], labels: dict[int, np.ndarray]
) -> dict | None:
    """Where the observation ray at ``frame`` meets the ball-centre court plane."""
    key = int(round(frame))
    if key not in cameras or key not in labels:
        return None
    try:
        point = whole.ground_point(cameras[key], labels[key])
    except (ValueError, np.linalg.LinAlgError):
        return None
    return {
        "frame": float(key),
        "court_xy_m": [float(point[0]), float(point[1])],
        "distance_to_net_plane_m": float(abs(point[1] - NET_PLANE_Y_M)),
        "inside_net_posts": bool(abs(point[0] - NET_CENTRE_X_M) <= NET_HALF_WIDTH_M),
    }


def modeled_height_m(frame: float, dense_flights: list[dict]) -> float | None:
    """Sample the fitted connected path's centre height at a native frame."""
    for flight in dense_flights:
        start, end = float(flight["start_frame"]), float(flight["end_frame"])
        if not start <= frame <= end:
            continue
        xyz = np.asarray(flight["positions"], float)
        if len(xyz) < 2:
            return float(xyz[0][2]) if len(xyz) else None
        times = np.linspace(start, end, len(xyz))
        return float(np.interp(frame, times, xyz[:, 2]))
    return None


def image_turn(
    frame: float, labels: dict[int, np.ndarray], span: int = 3
) -> tuple[float | None, dict]:
    """Angle between the observed image directions entering and leaving a frame."""
    key = int(round(frame))
    before = [labels[f] for f in range(key - span, key + 1) if f in labels]
    after = [labels[f] for f in range(key, key + span + 1) if f in labels]
    if len(before) < 2 or len(after) < 2:
        return None, {"reason": "insufficient_native_fronts_for_a_secant"}
    incoming = np.asarray(before[-1]) - np.asarray(before[0])
    outgoing = np.asarray(after[-1]) - np.asarray(after[0])
    norms = (float(np.linalg.norm(incoming)), float(np.linalg.norm(outgoing)))
    if min(norms) < 1e-6:
        return None, {"reason": "stationary_native_fronts"}
    cosine = float(np.dot(incoming, outgoing) / (norms[0] * norms[1]))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine)))), {
        "incoming_px": incoming.tolist(),
        "outgoing_px": outgoing.tolist(),
        "secant_span_frames": span,
    }


def _player_distance_m(court_xy: list[float] | None, players: list[dict]) -> float | None:
    if court_xy is None or not players:
        return None
    # Players whose court position is absent contribute no distance; with none
    # left this witness abstains instead of reporting a nearest player it has not
    # located.
    rooted = [row for row in players if player_position.root_xy(row) is not None]
    if not rooted:
        return None
    centres = np.asarray([row["court_centre_xy_m"] for row in rooted], float)
    return float(np.min(np.linalg.norm(centres - np.asarray(court_xy, float), axis=1)))


def kind_likelihoods(
    run: dict,
    cameras: dict[int, np.ndarray],
    labels: dict[int, np.ndarray],
    dense_flights: list[dict],
    players: list[dict],
    last_supported_frame: float,
) -> tuple[dict[str, float], dict]:
    """Score each event kind that could explain one unexplained span.

    The scores are explicit relative weights over the four physical kinds, not
    calibrated probabilities.  They exist so the branch order and the reported
    alternatives are reproducible, never to justify picking one silently.
    """
    onset = float(np.mean(run["onset_bracket_frames"]))
    ground = ground_ray_state(onset, cameras, labels)
    height = modeled_height_m(onset, dense_flights)
    turn, turn_evidence = image_turn(onset, labels)
    distance = _player_distance_m(None if ground is None else ground["court_xy_m"], players)
    weights = dict.fromkeys(KINDS, 0.0)

    turning = 0.0 if turn is None else min(1.0, turn / 45.0)
    low = height is not None and height <= GROUND_RAY_HEIGHT_M
    near_net = (
        ground is not None
        and ground["distance_to_net_plane_m"] <= NET_PLANE_TOLERANCE_M
        and ground["inside_net_posts"]
    )
    near_player = distance is not None and distance <= CONTACT_PLAYER_RADIUS_M

    weights["bounce"] = (0.6 if low else 0.15) + 0.4 * turning
    weights["contact"] = (0.6 if near_player else 0.15) + 0.4 * turning
    weights["net_hit"] = (0.6 if near_net else 0.05) + 0.2 * turning
    weights["ending"] = 0.7 if run["reaches_last_training_picture"] else 0.05
    if run["end_frame"] >= last_supported_frame - 1.0:
        weights["ending"] += 0.3
    total = sum(weights.values()) or 1.0
    normalized = {kind: float(value / total) for kind, value in weights.items()}
    evidence = {
        "onset_frame": onset,
        "modeled_centre_height_m": height,
        "ground_ray": ground,
        "image_turn_degrees": turn,
        "image_turn": turn_evidence,
        "nearest_automatic_player_distance_m": distance,
        "controls": {
            "ground_ray_height_m": GROUND_RAY_HEIGHT_M,
            "net_plane_tolerance_m": NET_PLANE_TOLERANCE_M,
            "contact_player_radius_m": CONTACT_PLAYER_RADIUS_M,
        },
    }
    return normalized, evidence


def propose(
    measurement: dict,
    cameras: dict[int, np.ndarray],
    labels: dict[int, np.ndarray],
    players: list[dict],
    last_supported_frame: float,
) -> list[Proposal]:
    """Turn every unexplained training span into one timed, kinded proposal."""
    proposals = []
    for run in residual_runs(measurement["native_projection"]):
        weights, evidence = kind_likelihoods(
            run, cameras, labels, measurement["dense_flights"], players, last_supported_frame
        )
        low, high = run["onset_bracket_frames"]
        proposals.append(
            Proposal(
                frame=float(np.mean([low, high])),
                frame_interval=(float(low), float(high)),
                kind_likelihoods=weights,
                evidence={**evidence, "residual_run": run},
            )
        )
    return proposals


def unwitnessable_supplied_events(
    events: list[dict],
    cameras: dict[int, np.ndarray],
    labels: dict[int, np.ndarray],
    *,
    contact_bracket_frames: int = 1,
) -> list[dict]:
    """Supplied events the observations cannot witness at all.

    This is the concrete reading of "the physics cannot explain it": a bounce
    whose whole interval is unlabelled has no native ground ray to compare a
    modelled impact against, and a contact whose bracket has no nearby visible
    front cannot be associated with a player.  Naming those events lets the
    search try demoting the one that actually blocked it, instead of walking the
    supplied topology in name order.  Nothing here reads a fitted result.
    """
    flagged = []
    for row in events:
        low, high = row.get("frame_interval", [row["frame"], row["frame"]])
        if row["event_type"] == "bounce":
            frames = {
                frame
                for frame in (
                    int(np.floor(low)),
                    int(np.ceil(low)),
                    int(np.floor(high)),
                    int(np.ceil(high)),
                )
                if frame in cameras and frame in labels
            }
            if not frames:
                flagged.append(
                    dict(row, demotion_reason="bounce_interval_has_no_visible_native_ground_ray")
                )
        elif row["event_type"] == "contact":
            window = range(
                int(np.floor(low)) - contact_bracket_frames,
                int(np.ceil(high)) + contact_bracket_frames + 1,
            )
            if not any(frame in labels for frame in window):
                flagged.append(
                    dict(row, demotion_reason="contact_bracket_has_no_nearby_visible_front")
                )
    return flagged


def _sorted_events(events: list[dict]) -> list[dict]:
    return sorted(events, key=lambda row: (float(row["frame"]), row["event_type"]))


def grammar_penalty(events: list[dict]) -> tuple[float, list[str]]:
    """Score a whole candidate topology against ordinary singles point grammar."""
    rows = _sorted_events(events)
    contacts = [row for row in rows if row["event_type"] == "contact"]
    bounces = [row for row in rows if row["event_type"] == "bounce"]
    endings = [row for row in rows if row["event_type"] == "ending"]
    penalty = 0.0
    reasons: list[str] = []
    if not contacts:
        return float("inf"), ["no_contact_topology"]
    if len(endings) != 1 or float(endings[0]["frame"]) < float(contacts[-1]["frame"]):
        return float("inf"), ["requires_one_post_contact_ending"]
    bounds = [float(row["frame"]) for row in contacts] + [float(endings[0]["frame"])]
    for index, (left, right) in enumerate(zip(bounds, bounds[1:])):
        count = sum(left < float(row["frame"]) <= right for row in bounces)
        terminal = index == len(bounds) - 2
        if terminal:
            if count not in {1, 2}:
                return float("inf"), [f"terminal_flight_has_{count}_bounces"]
            if count == 2:
                penalty += GRAMMAR_PENALTIES["second_bounce_before_the_ending"]
        elif count == 0:
            penalty += GRAMMAR_PENALTIES["volley_contact_without_preceding_bounce"]
            reasons.append(f"flight_{index}_is_a_volley")
        elif count > 1:
            return float("inf"), [f"flight_{index}_has_{count}_bounces"]
    nets = [row for row in rows if row["event_type"] == "net_hit"]
    penalty += len(nets) * GRAMMAR_PENALTIES["net_hit_inside_one_flight"]
    return penalty, reasons


def _event_row(kind: str, proposal: Proposal) -> dict:
    return {
        "event_type": kind,
        "frame": float(proposal.frame),
        "frame_interval": [float(proposal.frame_interval[0]), float(proposal.frame_interval[1])],
        "note": "recovered event hypothesis; input-only proposal, not a human label",
        "annotation_origin": "recovered",
        "kind_likelihood": float(proposal.kind_likelihoods[kind]),
    }


def topology_hypotheses(
    supplied: list[dict],
    proposals: list[Proposal],
    *,
    max_branches: int = 6,
    demotable: list[dict] | None = None,
    extensions: list[Hypothesis] | None = None,
) -> list[Hypothesis]:
    """Enumerate every topology branch the search will fit, best-first.

    The supplied topology is always branch zero, so recovery can only add
    alternatives; it can never remove the arm the reference already reports.

    ``extensions`` are structural readings the caller already justified from the
    observations -- today, the legal readings of a point whose modeled window
    ended at the net.  They lead the beam for the same reason an evidence-backed
    demotion does: their budget belongs to the event that actually blocked the
    fit, not to a residual run of a fit that never ran.
    """
    if max_branches < 1:
        raise ValueError("at least one topology branch required")
    penalty, reasons = grammar_penalty(supplied)
    branches = [
        Hypothesis(
            name="supplied",
            events=_sorted_events(supplied),
            grammar_penalty=0.0 if math.isinf(penalty) else penalty,
            reasons=["supplied topology, unchanged"] + reasons,
        )
    ]
    scored: list[tuple[float, Hypothesis]] = []
    for hypothesis in extensions or []:
        scored.append((0, hypothesis.grammar_penalty, hypothesis))
    for index, proposal in enumerate(proposals):
        for kind in proposal.kinds():
            row = _event_row(kind, proposal)
            events = _sorted_events([*supplied, row])
            value, why = grammar_penalty(events)
            if math.isinf(value):
                continue
            # Both kinds of branch carry the same base charge; a proposal
            # additionally pays for how unsure its kind is.  Scoring an add
            # without its charge made every proposal outrank every demotion
            # and crowded the spurious-event demotions out of the beam.
            score = (
                value
                + GRAMMAR_PENALTIES["add_a_recovered_event"]
                - math.log(max(proposal.kind_likelihoods[kind], 1e-6))
            )
            scored.append(
                (
                    1,
                    score,
                    Hypothesis(
                        name=f"add_{kind}_at_{proposal.frame:g}",
                        events=events,
                        added=[row],
                        grammar_penalty=value + GRAMMAR_PENALTIES["add_a_recovered_event"],
                        reasons=[
                            f"unexplained training residual run {index}",
                            *why,
                        ],
                    ),
                )
            )
    for row in demotable or []:
        # Match on kind and epoch, not object identity: an evidence-backed
        # candidate is a copy of the supplied row carrying its demotion reason,
        # so identity would silently leave the topology unchanged.
        target = (row["event_type"], float(row["frame"]))
        events = _sorted_events(
            [item for item in supplied if (item["event_type"], float(item["frame"])) != target]
        )
        if len(events) == len(supplied):
            continue
        value, why = grammar_penalty(events)
        if math.isinf(value):
            continue
        reason = row.get("demotion_reason")
        demoted = dict(row, demotion_reason=reason or "unexplained_by_physics")
        scored.append(
            (
                # A demotion the observations already justify leads the beam, so
                # its budget goes to the event that actually blocked the fit.
                # An ordinary demotion stays interleaved with the proposals and
                # competes on cost: pushing it behind every proposal crowded the
                # spurious-event demotions out of a four-branch beam.
                0 if reason else 1,
                value + GRAMMAR_PENALTIES["demote_a_supplied_event"],
                Hypothesis(
                    name=f"demote_{row['event_type']}_at_{float(row['frame']):g}",
                    events=events,
                    demoted=[demoted],
                    grammar_penalty=value + GRAMMAR_PENALTIES["demote_a_supplied_event"],
                    reasons=[demoted["demotion_reason"], *why],
                ),
            )
        )
    scored.sort(key=lambda item: (item[0], item[1], item[2].name))
    for *_, hypothesis in scored:
        if len(branches) >= max_branches:
            break
        branches.append(hypothesis)
    return branches


def match_recovered_events(
    recovered: list[dict], removed: list[dict], tolerance_frames: float = 3.0
) -> dict:
    """Score recovered events against the events a variant removed.

    Evaluation only.  The matcher runs after a branch was already selected on
    input-only evidence; the removed rows never reach the search.
    """
    remaining = list(removed)
    matches = []
    for row in sorted(recovered, key=lambda item: float(item["frame"])):
        candidates = [
            other
            for other in remaining
            if other["event_type"] == row["event_type"]
            and abs(float(other["frame"]) - float(row["frame"])) <= tolerance_frames
        ]
        if not candidates:
            matches.append({"recovered": row, "truth": None, "timing_error_frames": None})
            continue
        best = min(candidates, key=lambda other: abs(float(other["frame"]) - float(row["frame"])))
        remaining.remove(best)
        matches.append(
            {
                "recovered": row,
                "truth": best,
                "timing_error_frames": float(row["frame"]) - float(best["frame"]),
            }
        )
    true_positive = sum(1 for row in matches if row["truth"] is not None)
    errors = [abs(row["timing_error_frames"]) for row in matches if row["truth"] is not None]
    return {
        "matches": matches,
        "unrecovered": remaining,
        "true_positives": true_positive,
        "false_positives": len(matches) - true_positive,
        "false_negatives": len(remaining),
        "precision": None if not matches else true_positive / len(matches),
        "recall": None if not removed else true_positive / len(removed),
        "mean_absolute_timing_error_frames": None if not errors else float(np.mean(errors)),
        "maximum_absolute_timing_error_frames": None if not errors else float(np.max(errors)),
        "match_tolerance_frames": tolerance_frames,
    }


def propose_from_observations(
    cameras: dict[int, np.ndarray],
    labels: dict[int, np.ndarray],
    players: list[dict],
    supplied: list[dict],
    window: tuple[float, float],
    *,
    turn_degrees: float = 25.0,
    separation_frames: float = 3.0,
) -> list[Proposal]:
    """Propose events from the native fronts alone, with no fit available.

    A topology the current physical arm cannot even build - a flight with no
    bounce, or two bounces before an ordinary contact - has no residual to read.
    The observed image path still turns where a physical event happened, so this
    is the bootstrap the residual reader needs.  It reads only supplied native
    fronts, supplied cameras and automatic player positions.
    """
    low, high = float(window[0]), float(window[1])
    occupied = [float(row["frame"]) for row in supplied]
    turns = []
    for frame in sorted(labels):
        if not low + 1 <= frame <= high - 1:
            continue
        angle, evidence = image_turn(float(frame), labels)
        if angle is None or angle < turn_degrees:
            continue
        if any(abs(frame - value) < separation_frames for value in occupied):
            continue
        turns.append((float(frame), angle, evidence))
    peaks = [
        row
        for index, row in enumerate(turns)
        if all(
            abs(row[0] - other[0]) >= separation_frames or row[1] >= other[1]
            for position, other in enumerate(turns)
            if position != index
        )
    ]
    proposals = []
    for frame, angle, evidence in peaks:
        ground = ground_ray_state(frame, cameras, labels)
        distance = _player_distance_m(None if ground is None else ground["court_xy_m"], players)
        near_net = (
            ground is not None
            and ground["distance_to_net_plane_m"] <= NET_PLANE_TOLERANCE_M
            and ground["inside_net_posts"]
        )
        near_player = distance is not None and distance <= CONTACT_PLAYER_RADIUS_M
        turning = min(1.0, angle / 45.0)
        weights = {
            "bounce": 0.5 + 0.4 * turning - (0.3 if near_player else 0.0),
            "contact": (0.6 if near_player else 0.2) + 0.4 * turning,
            "net_hit": (0.6 if near_net else 0.05) + 0.2 * turning,
            "ending": 0.3 if frame >= high - 2 else 0.02,
        }
        weights = {kind: max(value, 0.0) for kind, value in weights.items()}
        total = sum(weights.values()) or 1.0
        proposals.append(
            Proposal(
                frame=frame,
                frame_interval=(frame - 1.0, frame + 1.0),
                kind_likelihoods={kind: float(value / total) for kind, value in weights.items()},
                evidence={
                    "image_turn_degrees": angle,
                    "image_turn": evidence,
                    "ground_ray": ground,
                    "nearest_automatic_player_distance_m": distance,
                    "controls": {
                        "turn_degrees": turn_degrees,
                        "separation_frames": separation_frames,
                    },
                },
                source="native_front_direction_turn",
            )
        )
    return sorted(proposals, key=lambda row: row.frame)


def ending_candidates(supplied: list[dict], window: tuple[float, float]) -> list[Proposal]:
    """Propose where a missing point ending could be, from grammar alone.

    An ending is not a picture-level turn: it is the epoch after which no
    further ordinary contact happens.  The only input-only candidates are the
    supplied post-contact bounces and the last supported native picture.
    """
    contacts = [float(row["frame"]) for row in supplied if row["event_type"] == "contact"]
    if not contacts:
        return []
    last_contact = max(contacts)
    frames = sorted(
        {
            float(row["frame"])
            for row in supplied
            if row["event_type"] == "bounce" and float(row["frame"]) > last_contact
        }
        | {float(window[1])}
    )
    return [
        Proposal(
            frame=frame,
            frame_interval=(frame, frame),
            kind_likelihoods={"ending": 1.0, "contact": 0.0, "bounce": 0.0, "net_hit": 0.0},
            evidence={
                "last_supplied_contact_frame": last_contact,
                "last_supported_native_frame": float(window[1]),
                "rule": "post-contact bounce epochs and the last supported picture",
            },
            source="grammar_ending_candidate",
        )
        for frame in frames
    ]


def _terminal_flight_bounce_count(events: list[dict]) -> int | None:
    """Bounces between the last contact and the ending, or ``None`` if malformed."""
    contacts = [float(row["frame"]) for row in events if row["event_type"] == "contact"]
    endings = [float(row["frame"]) for row in events if row["event_type"] == "ending"]
    if not contacts or len(endings) != 1 or endings[0] < max(contacts):
        return None
    return sum(
        max(contacts) < float(row["frame"]) <= endings[0]
        for row in events
        if row["event_type"] == "bounce"
    )


def vertical_reversal(
    frame: float, labels: dict[int, np.ndarray], span: int = TERMINAL_REVERSAL_BRACKET_FRAMES
) -> dict | None:
    """Whether the native front's downward image motion reverses at ``frame``.

    A ball that reaches the court reverses its image row: it is descending in
    the pictures before the impact and rising in the pictures after it.  This is
    an input-only witness read from the same visible fronts the objective uses,
    with no fitted state and no evaluation label.
    """
    key = int(round(frame))
    before = [(f, labels[f]) for f in range(key - span, key + 1) if f in labels]
    after = [(f, labels[f]) for f in range(key, key + span + 1) if f in labels]
    if len(before) < 2 or len(after) < 2:
        return None
    incoming = float(before[-1][1][1] - before[0][1][1]) / (before[-1][0] - before[0][0])
    outgoing = float(after[-1][1][1] - after[0][1][1]) / (after[-1][0] - after[0][0])
    return {
        "frame": float(key),
        "incoming_image_row_rate_px_per_frame": incoming,
        "outgoing_image_row_rate_px_per_frame": outgoing,
        "reverses_downward_motion": bool(incoming > 0.0 > outgoing),
        "bracket_frames": span,
    }


def net_termination_hypotheses(
    supplied: list[dict],
    dead_ball_events: list[dict],
    labels: dict[int, np.ndarray],
    uncertain_events: list[dict] | None = None,
) -> list[Hypothesis]:
    """Legal readings of a point whose modeled window ends with no ground bounce.

    Every attempt this answers is one the connected grammar refuses outright: the
    terminal flight has no bounce because the ending was placed at the net, so
    the supplied topology can never be built and the attempt is lost before the
    fitter runs.  The physical reading is that the ball kept moving after the
    net and reached the court, and the labels usually say where.  Three readings
    are enumerated, each with the evidence that licenses it:

    * the annotator labeled a dead-ball bounce after the ending -- extend the
      modeled window to it and keep the net hit inside the terminal flight;
    * the terminal epoch is itself ambiguous between mesh and ground and the
      native fronts reverse there -- read it as a ground impact;
    * neither holds -- end at the last mid-rally bounce, which drops the final
      labeled contact and is reported as a shortened topology, never as a
      reconstruction of the labeled point.

    Nothing here reads a fitted result, a withheld picture or an evaluation
    label, and the supplied topology is never removed from the beam.
    """
    if _terminal_flight_bounce_count(supplied) != 0:
        return []
    contacts = [float(row["frame"]) for row in supplied if row["event_type"] == "contact"]
    ending = next(row for row in supplied if row["event_type"] == "ending")
    ending_frame = float(ending["frame"])
    last_contact = max(contacts)
    physical = [row for row in supplied if row["event_type"] != "ending"]
    branches: list[Hypothesis] = []

    # The dead-ball impact may already sit in the supplied topology, outside the
    # modeled window; then only the ending moves and no row is added.
    supplied_after = [
        row
        for row in physical
        if row["event_type"] == "bounce" and float(row["frame"]) >= ending_frame
    ]
    known = {float(row["frame"]) for row in supplied_after}
    trailing = sorted(
        [
            *supplied_after,
            *(
                row
                for row in dead_ball_events
                if float(row["frame"]) >= ending_frame and float(row["frame"]) not in known
            ),
        ],
        key=lambda row: float(row["frame"]),
    )
    for row in trailing[:1]:
        frame = float(row["frame"])
        already_supplied = frame in known
        added = {
            "event_type": "bounce",
            "frame": frame,
            "frame_interval": [float(value) for value in row.get("frame_interval", [frame, frame])],
            "note": "labeled dead-ball ground impact taken as the terminal bounce",
            "annotation_origin": "recovered",
            "supplied_status": row.get("status"),
        }
        moved = dict(ending, frame=frame, frame_interval=added["frame_interval"])
        events = _sorted_events(
            [*physical, moved] if already_supplied else [*physical, added, moved]
        )
        value, why = grammar_penalty(events)
        if math.isinf(value):
            continue
        branches.append(
            Hypothesis(
                name=f"extend_to_dead_ball_bounce_at_{frame:g}",
                events=events,
                added=[added],
                grammar_penalty=value + GRAMMAR_PENALTIES["extend_to_a_labeled_dead_ball_bounce"],
                reasons=[
                    "the modeled window ended at the net with no ground bounce",
                    f"labeled dead-ball ground impact at {frame:g} taken as the terminal bounce",
                    f"modeled window extended from {ending_frame:g} to {frame:g}",
                    *why,
                ],
            )
        )

    if not branches:
        witness = vertical_reversal(ending_frame, labels)
        if witness is not None and witness["reverses_downward_motion"]:
            added = {
                "event_type": "bounce",
                "frame": ending_frame,
                "frame_interval": [
                    float(value) for value in ending.get("frame_interval", [ending_frame] * 2)
                ],
                "note": "terminal epoch read as a ground impact; native fronts reverse there",
                "annotation_origin": "recovered",
                "image_reversal_witness": witness,
            }
            events = _sorted_events([*supplied, added])
            value, why = grammar_penalty(events)
            if not math.isinf(value):
                branches.append(
                    Hypothesis(
                        name=f"terminal_ground_impact_at_{ending_frame:g}",
                        events=events,
                        added=[added],
                        grammar_penalty=(
                            value + GRAMMAR_PENALTIES["terminal_ground_impact_at_the_ending"]
                        ),
                        reasons=[
                            "the modeled window ended at the net with no ground bounce",
                            "no labeled dead-ball bounce follows the ending",
                            "native fronts reverse their downward image motion at the ending",
                            *why,
                        ],
                    )
                )

    # The cut may also fall on a bounce the annotator recorded but would not
    # certify.  An abstention on certainty is not an absent impact, and this is
    # the last branch before the attempt is lost outright, so the uncertain row
    # is admitted here -- and only here -- with its status carried through.
    uncertain = [
        row
        for row in uncertain_events or []
        if row["event_type"] == "bounce"
        and float(row["frame"]) < last_contact
        and float(row["frame"]) > min(contacts)
    ]
    earlier = [
        float(row["frame"])
        for row in supplied
        if row["event_type"] == "bounce" and float(row["frame"]) < last_contact
    ]
    uncertain_cut = sorted(float(row["frame"]) for row in uncertain)
    if not earlier and uncertain_cut:
        earlier = [uncertain_cut[-1]]
    if earlier and len(contacts) > 1:
        cut = max(earlier)
        kept = [row for row in physical if float(row["frame"]) <= cut]
        dropped = [row for row in physical if float(row["frame"]) > cut]
        if cut in set(uncertain_cut) and not any(
            row["event_type"] == "bounce" and float(row["frame"]) == cut for row in kept
        ):
            source = next(row for row in uncertain if float(row["frame"]) == cut)
            kept = [
                *kept,
                {
                    "event_type": "bounce",
                    "frame": cut,
                    "frame_interval": [
                        float(value) for value in source.get("frame_interval", [cut, cut])
                    ],
                    "note": "uncertain-status ground impact admitted as the truncation boundary",
                    "annotation_origin": "recovered",
                    "supplied_status": source.get("status"),
                },
            ]
        events = _sorted_events([*kept, dict(ending, frame=cut, frame_interval=[cut, cut])])
        value, why = grammar_penalty(events)
        if not math.isinf(value):
            branches.append(
                Hypothesis(
                    name=f"truncate_to_bounce_at_{cut:g}",
                    events=events,
                    demoted=[
                        dict(row, demotion_reason="after_the_last_reachable_terminal_bounce")
                        for row in dropped
                    ],
                    grammar_penalty=(
                        value + GRAMMAR_PENALTIES["truncate_to_the_last_mid_rally_bounce"]
                    ),
                    reasons=[
                        "the modeled window ended at the net with no ground bounce",
                        f"shortened topology: {len(dropped)} labeled event(s) after {cut:g} dropped",
                        "this is a partial point, not a reconstruction of the labeled topology",
                        *why,
                    ],
                )
            )
    return branches


def extend_attempt_window(attempt: dict, ball_records: list[dict], end_frame: float) -> dict:
    """Copy an attempt with its native label inventory carried to ``end_frame``.

    A recovered topology that ends later than the supplied one needs the fronts
    between the two epochs, and those rows are already in the frozen label
    document.  Nothing is invented: the rows are copied verbatim, the inventory
    must stay contiguous, and the returned attempt is used only by the branch
    that asked for it.
    """
    current = float(attempt["owner_end_frame"])
    if end_frame <= current:
        return attempt
    window = [row for row in ball_records if row["clip"] == attempt["point_clip"]]
    if len(window) != 1:
        raise ValueError("one matching frozen ball window required to extend the modeled window")
    rows = {int(row["frame"]): row for row in window[0]["frames"]}
    wanted = list(range(int(np.floor(current)) + 1, int(np.floor(end_frame)) + 1))
    if any(frame not in rows for frame in wanted):
        raise ValueError("contiguous native inventory required to extend the modeled window")
    added = [rows[frame] for frame in wanted]
    extended = [*attempt["owner_ball_labels"], *added]
    return {
        **attempt,
        "owner_ball_labels": extended,
        "owner_end_frame": float(end_frame),
        "labeled_native_frames": len(extended),
        "visible_native_frames": sum(row["status"] == "visible" for row in extended),
        "context_native_frames": [
            frame for frame in attempt["context_native_frames"] if frame not in set(wanted)
        ],
        "modeled_window_extension": {
            "from_frame": current,
            "to_frame": float(end_frame),
            "native_frames_added": wanted,
            "source": "frozen label document; rows copied verbatim",
        },
    }
