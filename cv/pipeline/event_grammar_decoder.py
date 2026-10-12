"""Linear-chain Viterbi decoder over the point's event grammar.

The classifier decoder this replaces takes the per-frame argmax and suppresses
neighbours.  Here every plausible (frame, type) pair becomes a node of a
per-point lattice, transitions carry the grammar of a rally

    serve -> contact -> bounce | net_hit -> contact -> ... -> end

with side alternation between successive contacts, at most one bounce between
two contacts, and a minimum flight duration between events.  The best path is
the emission; running sum-product over the same lattice gives each node's
posterior marginal, and the gap between the best path (which asserts the node
with probability one) and that marginal is the abstention score.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from cv.pipeline import event_time_distribution as timing
from cv.pipeline import event_time_neighborhood as neighborhood
from cv.pipeline import point_context as point_context_module

CLASSES = ("none", "contact", "bounce", "net_hit")
EVENT_TYPES = ("contact", "bounce", "net_hit")

NODE_FLOOR = 0.01
# Per-type overrides of the lattice floor.  ``net_hit`` is 3.8% of the truth and
# its probabilities live an order of magnitude below the other two types, so it
# needs its own floor to become a node at all.
TYPE_FLOORS = {"contact": NODE_FLOOR, "bounce": NODE_FLOOR, "net_hit": NODE_FLOOR}
# Per-type additive gain, in log odds, applied to every node of that type.
TYPE_LOG_BONUS = {"contact": 0.0, "bounce": 0.0, "net_hit": 0.0}
LOCAL_MAXIMUM_RADIUS = 2
NEGATIVE_INFINITY = -1e30

# Label-free proposal windows (``cv.pipeline.event_proposals``) are evidence
# about a node, not a node.  ``proposal_evidence`` turns a proposal document into
# a per-row, per-type additive log-prior that ``build_nodes`` adds to the node's
# classifier gain, bounded by ``PROPOSAL_PRIOR_CAP`` so a confident proposal can
# move a node inside the grammar's own dynamic range and never replace it: with
# the cap at one nat a proposal is worth less than the contact-to-contact
# transition prior and far less than the side-alternation penalty.  Proposal
# strength is the proposer's own ``confidence``, which for ``physics_departure``
# is ``min(1, departure_sigma / 2k)`` -- monotone in the departure sigma the
# arm reports.  Several proposals on one row take the maximum, never a sum, so a
# dense arm cannot buy log-odds by repeating itself.
PROPOSAL_PRIOR_WEIGHT = 0.0
PROPOSAL_PRIOR_CAP = 1.0
# A proposal may also admit a node the classifier scores above the type floor but
# which is not a local maximum, *only* on a frame the composed track does not
# observe.  On an observed frame the track already speaks and the classifier's
# local maximum stands.
PROPOSAL_ADMISSION_STRENGTH = 0.5
# How a proposal's strength is read.  ``confidence`` takes the proposer's own
# number.  For ``physics_departure`` that is ``min(1, departure_sigma / 2k)``,
# which **saturates**: on the current cohort a third of all proposals sit at
# exactly 1.0, so a 69-sigma departure and a 7-sigma one are given identical
# weight and the prior stops discriminating.  ``sigma`` reads the departure sigma
# itself on an unsaturated log scale against ``PROPOSAL_SIGMA_REFERENCE``, and
# falls back to the proposer's confidence for the arms that report no sigma.
PROPOSAL_STRENGTH_CONFIDENCE = "confidence"
PROPOSAL_STRENGTH_SIGMA = "sigma"
PROPOSAL_STRENGTH_MODES = (PROPOSAL_STRENGTH_CONFIDENCE, PROPOSAL_STRENGTH_SIGMA)
PROPOSAL_SIGMA_REFERENCE = 30.0
# Optional local contrast: subtract the median strength of the same clip and type
# inside +-``local_baseline_frames`` exposures.  The proposal union covers most
# active-play frames, so an uncorrected prior is close to a constant offset that
# moves every marginal and almost no path; the contrast keeps only evidence that
# stands out where it sits.
PROPOSAL_LOCAL_BASELINE_FRAMES = 0

# How the decoder turns one lattice into emitted rows.  ``best_path`` is the
# shipped behaviour: only the Viterbi argmax path is emitted.  ``path_marginal``
# additionally emits every node whose sum-product posterior clears
# ``OFF_PATH_MARGINAL_FLOOR``, so a node the argmax path skips can still be
# accepted on its own posterior.  ``nbest_union`` emits the union of the ``k``
# best paths and scores each node by the fraction of those paths that contain it.
BEST_PATH = "best_path"
PATH_MARGINAL = "path_marginal"
NBEST_UNION = "nbest_union"
EMISSION_MODES = (BEST_PATH, PATH_MARGINAL, NBEST_UNION)
OFF_PATH_MARGINAL_FLOOR = 0.02
NBEST_PATHS = 8

# Minimum frames between two events, by (previous type, next type).
MINIMUM_GAP = {
    ("contact", "bounce"): 3,
    ("contact", "net_hit"): 2,
    ("contact", "contact"): 6,
    ("bounce", "contact"): 3,
    ("bounce", "net_hit"): 3,
    ("net_hit", "bounce"): 2,
    ("net_hit", "contact"): 3,
}
# Log-prior for each allowed transition; missing pairs are forbidden.
TRANSITION_LOG_PRIOR = {
    ("contact", "bounce"): 0.0,
    ("contact", "net_hit"): -1.5,
    ("contact", "contact"): -1.5,
    ("bounce", "contact"): 0.0,
    ("bounce", "net_hit"): -2.5,
    ("net_hit", "bounce"): -0.5,
    ("net_hit", "contact"): -0.5,
}
START_LOG_PRIOR = {"contact": 0.0, "bounce": -1.0, "net_hit": -2.0}
# Optional score for ending a path on each physical kind.  Zero reproduces the
# shipped decoder.  Candidate-path decoding may price a plausible bounce/net
# ending separately from the same observation used inside a rally.
END_LOG_PRIOR = {"contact": 0.0, "bounce": 0.0, "net_hit": 0.0}
SIDE_VIOLATION_PENALTY = -3.0
# The court coordinate ``court_y`` the feature builder carries is not the
# fraction of the court between the two baselines.  Traced to its root
# (``cv/pipeline/event_model_v2_features`` module docstring): the builder reads
# the track's legacy 960x540 ``x``/``y`` columns and pushes them through a
# native-1920x1080 homography, so ``court_y`` is the ground-plane projection of a
# half-resolution pixel.  On the frozen cohort it runs from about 0.4 to 3.2, the
# net sits near 1.29 and its position differs per point with the camera.
# ``NET_LINE = 0.5`` therefore called 98.2% of all rows "far" and made the
# side-alternation penalty fire on 96.4% of genuinely alternating truth contact
# pairs.  See ``docs/wk1/grammar_path.md`` and the ``netline`` package report.
#
# The fix is not another constant.  ``net_line`` and ``side_dead_band`` accept a
# ``{clip: value}`` mapping, and
# ``cv.pipeline.event_model_v2_features.derive_net_lines`` builds one by pushing
# the net's own pixels -- its ground line from the point homography and its tape
# top from the camera bundle's observed net cord -- through the builder's own
# projection.  A clip with no derived value falls back to ``net_line_default``.
LEGACY_NET_LINE = 0.5
NET_LINE = LEGACY_NET_LINE
# Half-width of the band around ``net_line`` in which the side is called unknown.
SIDE_DEAD_BAND = 0.0
# Largest frame correction the time head is allowed to apply to an emission.
MAX_TIME_CORRECTION = 3.0
CROP_HALF_WIDTH = 96.0
CROP_HALF_HEIGHT = 54.0
# Court coordinates carried by the crop manifest are normalised against the
# outside doubles court.  A small tolerance keeps a line impact from becoming a
# spurious out call because the event xy head is not a line-calling model.
COURT_BOUNDARY_TOLERANCE = 0.02
TRACK_EXIT_EDGE_PX = 24.0
NATIVE_WIDTH = 1920.0
NATIVE_HEIGHT = 1080.0
ATTEMPT_RESET_GAP_FRAMES = 80.0


def _describe_side_value(value):
    """Summarise a scalar or per-point side constant for a run manifest."""

    if not isinstance(value, Mapping):
        return value
    numbers = sorted(float(item) for item in value.values())
    middle = len(numbers) // 2
    return {
        "kind": "per_point",
        "points": len(numbers),
        "min": numbers[0] if numbers else None,
        "median": (
            None
            if not numbers
            else numbers[middle]
            if len(numbers) % 2
            else 0.5 * (numbers[middle - 1] + numbers[middle])
        ),
        "max": numbers[-1] if numbers else None,
    }


@dataclass(frozen=True)
class GrammarConfig:
    """Every tunable of the decoder in one place.

    The defaults reproduce the first-pass decoder exactly; the benchmark sweeps
    variants of this object over saved probabilities without retraining.
    """

    floors: dict = None
    type_log_bonus: dict = None
    transition_log_prior: dict = None
    start_log_prior: dict = None
    end_log_prior: dict = None
    minimum_gap: dict = None
    side_violation_penalty: float = SIDE_VIOLATION_PENALTY
    local_maximum_radius: int = LOCAL_MAXIMUM_RADIUS
    net_line: float | dict | None = None
    side_dead_band: float | dict = SIDE_DEAD_BAND
    exact_side_state: bool = False
    net_line_default: float | None = None
    # A bounce after an in-court bounce is a physical event in the general
    # point grammar.  It happens to terminate a tennis attempt, but the ending
    # row is derived later by ``point_end_rows``; it is not an endpoint-only
    # classifier head.
    terminal_second_bounce: bool = False
    terminal_second_bounce_log_prior: float = -0.5
    # The first bounce that witnesses a recovered second bounce remains a
    # lattice/support state.  Do not let the availability of a terminal suffix
    # turn an otherwise abstained support state into an emitted physical event.
    # This is deliberately configurable for historical replay only.
    emit_terminal_support_bounce: bool = False
    # A high-confidence ordinary contact after a proposed second bounce is
    # direct automatic evidence that play continues.  Do not let a terminal
    # suffix replace that continuation merely because it raises a nearby bounce
    # posterior.
    terminal_requires_no_ordinary_contact: bool = True

    def resolved(self) -> "GrammarConfig":
        return GrammarConfig(
            floors=dict(self.floors or TYPE_FLOORS),
            type_log_bonus=dict(self.type_log_bonus or TYPE_LOG_BONUS),
            transition_log_prior=dict(self.transition_log_prior or TRANSITION_LOG_PRIOR),
            start_log_prior=dict(self.start_log_prior or START_LOG_PRIOR),
            end_log_prior=dict(self.end_log_prior or END_LOG_PRIOR),
            minimum_gap=dict(self.minimum_gap or MINIMUM_GAP),
            side_violation_penalty=self.side_violation_penalty,
            local_maximum_radius=self.local_maximum_radius,
            net_line=(
                NET_LINE
                if self.net_line is None
                else dict(self.net_line)
                if isinstance(self.net_line, Mapping)
                else float(self.net_line)
            ),
            side_dead_band=(
                dict(self.side_dead_band)
                if isinstance(self.side_dead_band, Mapping)
                else float(self.side_dead_band)
            ),
            exact_side_state=bool(self.exact_side_state),
            terminal_second_bounce=bool(self.terminal_second_bounce),
            terminal_second_bounce_log_prior=float(self.terminal_second_bounce_log_prior),
            emit_terminal_support_bounce=bool(self.emit_terminal_support_bounce),
            terminal_requires_no_ordinary_contact=bool(self.terminal_requires_no_ordinary_contact),
            net_line_default=(
                NET_LINE if self.net_line_default is None else float(self.net_line_default)
            ),
        )

    def for_clip(self, clip: str | None) -> "GrammarConfig":
        """Return this config with ``net_line``/``side_dead_band`` scalar for ``clip``.

        A per-point table is how the derived net line travels; a clip the table
        does not name falls back to ``net_line_default`` and to no dead band.
        """

        resolved = self.resolved()
        line = resolved.net_line
        band = resolved.side_dead_band
        if isinstance(line, Mapping):
            line = float(line.get(clip, resolved.net_line_default))
        if isinstance(band, Mapping):
            band = float(band.get(clip, 0.0))
        return replace(resolved, net_line=float(line), side_dead_band=float(band))

    def as_dict(self) -> dict:
        resolved = self.resolved()
        return {
            "floors": resolved.floors,
            "type_log_bonus": resolved.type_log_bonus,
            "transition_log_prior": {
                f"{left}->{right}": value
                for (left, right), value in resolved.transition_log_prior.items()
            },
            "start_log_prior": resolved.start_log_prior,
            "end_log_prior": resolved.end_log_prior,
            "minimum_gap": {
                f"{left}->{right}": value for (left, right), value in resolved.minimum_gap.items()
            },
            "side_violation_penalty": resolved.side_violation_penalty,
            "local_maximum_radius": resolved.local_maximum_radius,
            "net_line": _describe_side_value(resolved.net_line),
            "side_dead_band": _describe_side_value(resolved.side_dead_band),
            "net_line_default": resolved.net_line_default,
            "exact_side_state": resolved.exact_side_state,
            "terminal_second_bounce": resolved.terminal_second_bounce,
            "terminal_second_bounce_log_prior": resolved.terminal_second_bounce_log_prior,
            "emit_terminal_support_bounce": resolved.emit_terminal_support_bounce,
            "terminal_requires_no_ordinary_contact": resolved.terminal_requires_no_ordinary_contact,
        }


DEFAULT_GRAMMAR = GrammarConfig()


@dataclass(frozen=True)
class Node:
    index: int
    frame: int
    event_type: str
    gain: float
    side: int
    terminal: bool = False


def _logsumexp(values: list[float]) -> float:
    finite = [value for value in values if value > NEGATIVE_INFINITY / 2]
    if not finite:
        return NEGATIVE_INFINITY
    top = max(finite)
    return top + math.log(sum(math.exp(value - top) for value in finite))


def side_of(court_y: float, net_line: float | None = None, dead_band: float = 0.0) -> int:
    """Return ``+1``/``-1`` for the far/near half of the court, ``0`` if unknown.

    ``court_y`` is the ball's ground-plane projection divided by the court
    length, so the value of ``net_line`` that actually separates the two halves
    is a property of that projection and not of the geometry; it is calibrated,
    not assumed.  Inside ``dead_band`` of the line the side is called unknown, so
    the side-alternation penalty never fires on a coordinate too close to the net
    to trust.
    """

    if court_y is None or not math.isfinite(court_y):
        return 0
    line = NET_LINE if net_line is None else float(net_line)
    if dead_band and abs(court_y - line) < dead_band:
        return 0
    return 1 if court_y > line else -1


def build_nodes(
    probabilities: np.ndarray,
    frames: np.ndarray,
    court_y: np.ndarray,
    *,
    config: GrammarConfig | None = None,
    clip: str | None = None,
    evidence_log_prior: np.ndarray | None = None,
    evidence_admits: np.ndarray | None = None,
    tentative_probability_floor: float | None = None,
    relax_local_maximum: bool = False,
) -> list[Node]:
    """Return the lattice nodes of one point, ordered by frame then type.

    A node exists where the type probability clears that type's floor and is a
    local maximum of the type inside ``local_maximum_radius`` frames, which keeps
    the lattice small without committing to the argmax the way the production
    decoder does.  Each type may also carry an additive log-odds bonus.

    ``evidence_log_prior`` is an optional ``(rows, len(EVENT_TYPES))`` array of
    bounded proposal log-priors added to the node gain.  It is deliberately not
    consulted by the floor or the local-maximum test: which rows may become
    nodes stays the classifier's decision.  ``evidence_admits`` is the one
    exception and is restricted by its builder to frames the composed track does
    not observe, where there is no track maximum to defer to.

    ``tentative_probability_floor`` and ``relax_local_maximum`` are an additive,
    default-off escape hatch for candidate export only.  They let downstream
    physics inspect classifier rows that the firm lattice deliberately suppresses;
    callers must not use them to build the firm decoder path.
    """

    config = (config or DEFAULT_GRAMMAR).for_clip(clip)
    radius = config.local_maximum_radius
    order = np.argsort(frames, kind="stable")
    nodes: list[Node] = []
    none_probability = np.clip(probabilities[:, 0], 1e-9, 1.0)
    for class_index, event_type in enumerate(EVENT_TYPES, start=1):
        column = probabilities[:, class_index]
        floor = (
            float(config.floors[event_type])
            if tentative_probability_floor is None
            else float(tentative_probability_floor)
        )
        bonus = float(config.type_log_bonus[event_type])
        admits = None if evidence_admits is None else evidence_admits[:, class_index - 1]
        prior = None if evidence_log_prior is None else evidence_log_prior[:, class_index - 1]
        for position in order:
            value = float(column[position])
            if value < floor:
                continue
            frame = int(frames[position])
            neighbours = [
                float(column[other])
                for other in order
                if abs(int(frames[other]) - frame) <= radius and other != position
            ]
            admitted = admits is not None and bool(admits[position])
            if not relax_local_maximum and not admitted and neighbours and value < max(neighbours):
                continue
            if (
                not relax_local_maximum
                and not admitted
                and neighbours
                and value == max(neighbours)
                and any(
                    float(column[other]) == value and int(frames[other]) < frame
                    for other in order
                    if abs(int(frames[other]) - frame) <= radius and other != position
                )
            ):
                continue
            nodes.append(
                Node(
                    index=int(position),
                    frame=frame,
                    event_type=event_type,
                    gain=float(
                        math.log(max(value, 1e-9))
                        - math.log(none_probability[position])
                        + bonus
                        + (0.0 if prior is None else float(prior[position]))
                    ),
                    side=side_of(float(court_y[position]), config.net_line, config.side_dead_band),
                )
            )
    if config.terminal_second_bounce:
        nodes.extend(
            replace(node, terminal=True) for node in tuple(nodes) if node.event_type == "bounce"
        )
    nodes.sort(key=lambda node: (node.frame, EVENT_TYPES.index(node.event_type), node.terminal))
    return nodes


def proposal_strength(proposal: Mapping, mode: str, sigma_reference: float) -> float:
    """Return one proposal's evidence strength in ``[0, 1]``.

    ``sigma`` mode reads ``departure_sigma`` -- the number the physics arm's
    prediction-interval test actually produces -- on ``log1p`` of its excess over
    the arm's own decision threshold ``departure_k``, normalised against
    ``sigma_reference``.  That keeps the strength monotone in the departure and
    unsaturated over the range the arm reports.  A proposal with no departure
    sigma (pose, corner, implied bounce) keeps its own confidence, which is the
    only statistic it has.
    """

    confidence = max(0.0, min(1.0, float(proposal.get("confidence", 0.0))))
    if mode == PROPOSAL_STRENGTH_CONFIDENCE:
        return confidence
    if mode != PROPOSAL_STRENGTH_SIGMA:
        raise ValueError(f"unknown proposal strength mode: {mode}")
    evidence = proposal.get("evidence") or {}
    sigma = evidence.get("departure_sigma")
    if sigma is None:
        return confidence
    threshold = float(evidence.get("departure_k", 0.0))
    if sigma_reference <= threshold:
        raise ValueError("the sigma reference must exceed the arm's own departure threshold")
    excess = max(0.0, float(sigma) - threshold)
    return min(1.0, math.log1p(excess) / math.log1p(sigma_reference - threshold))


def proposal_evidence(
    clips: np.ndarray,
    frames: np.ndarray,
    proposals,
    *,
    weight: float = PROPOSAL_PRIOR_WEIGHT,
    cap: float = PROPOSAL_PRIOR_CAP,
    track_observed: np.ndarray | None = None,
    admission_strength: float | None = None,
    sources: tuple[str, ...] | None = None,
    strength_mode: str = PROPOSAL_STRENGTH_CONFIDENCE,
    sigma_reference: float = PROPOSAL_SIGMA_REFERENCE,
    local_baseline_frames: int = PROPOSAL_LOCAL_BASELINE_FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn label-free proposal windows into a bounded per-row, per-type prior.

    ``proposals`` are the rows of a ``cv.pipeline.event_proposals`` document:
    each carries a clip, a ``[start_frame, end_frame]`` timing range, the physical
    ``kinds`` it is compatible with, and a ``confidence``.  A row of the lattice
    receives a proposal's strength when the proposal names its clip, its integer
    frame lies inside the proposal's own timing range -- the range is the
    proposer's stated uncertainty and is not widened here -- and the type is one
    the proposal is compatible with.  Overlapping proposals take the maximum, so
    the prior measures the best available evidence rather than the local window
    density.

    Returns ``(log_prior, admits)``.  ``log_prior`` is ``min(cap, weight *
    strength)`` per row and type.  ``admits`` is True only where
    ``track_observed`` is False and the strength clears ``admission_strength``;
    it is all False when either is not supplied, which is the default.
    """

    if weight < 0.0 or cap < 0.0:
        raise ValueError("the proposal prior weight and cap must be non-negative")
    rows = len(frames)
    strength = np.zeros((rows, len(EVENT_TYPES)), dtype=np.float64)
    positions: dict[str, list[int]] = defaultdict(list)
    for position, clip in enumerate(np.asarray(clips).tolist()):
        positions[str(clip)].append(position)
    integer_frames = np.asarray(frames).astype(np.int64)
    for proposal in proposals:
        if sources is not None and str(proposal.get("source")) not in sources:
            continue
        clip = str(proposal["clip"])
        members = positions.get(clip)
        if not members:
            continue
        value = proposal_strength(proposal, strength_mode, sigma_reference)
        if value <= 0.0:
            continue
        start = float(min(proposal["start_frame"], proposal["end_frame"]))
        end = float(max(proposal["start_frame"], proposal["end_frame"]))
        columns = [
            EVENT_TYPES.index(str(kind)) for kind in proposal["kinds"] if str(kind) in EVENT_TYPES
        ]
        if not columns:
            continue
        for position in members:
            frame = int(integer_frames[position])
            if not start <= frame <= end:
                continue
            for column in columns:
                if value > strength[position, column]:
                    strength[position, column] = value
    if local_baseline_frames:
        contrast = np.zeros_like(strength)
        for members in positions.values():
            order = sorted(members, key=lambda position: int(integer_frames[position]))
            ordered = np.asarray([int(integer_frames[position]) for position in order])
            block = strength[order]
            for offset, position in enumerate(order):
                window = np.abs(ordered - ordered[offset]) <= int(local_baseline_frames)
                contrast[position] = np.maximum(
                    0.0, strength[position] - np.median(block[window], axis=0)
                )
        strength = contrast
    log_prior = np.minimum(float(cap), float(weight) * strength)
    admits = np.zeros_like(strength, dtype=bool)
    if track_observed is not None and admission_strength is not None:
        unobserved = ~np.asarray(track_observed).astype(bool)
        admits = unobserved[:, None] & (strength >= float(admission_strength))
    return log_prior, admits


def _start_prior(node: Node, config: GrammarConfig) -> float:
    return NEGATIVE_INFINITY if node.terminal else config.start_log_prior[node.event_type]


def _transition_prior(previous: Node, current: Node, config: GrammarConfig) -> float:
    if previous.terminal:
        return NEGATIVE_INFINITY
    if current.terminal:
        # A second bounce closes the path. It cannot start a path or be followed
        # by a contact; it is still the same physical bounce observation/type.
        if (
            config.terminal_second_bounce
            and previous.event_type == "bounce"
            and current.frame - previous.frame >= 3
        ):
            return config.terminal_second_bounce_log_prior
        return NEGATIVE_INFINITY
    pair = (previous.event_type, current.event_type)
    if (
        pair not in config.transition_log_prior
        or current.frame - previous.frame < config.minimum_gap[pair]
    ):
        return NEGATIVE_INFINITY
    return config.transition_log_prior[pair]


def transition_score(
    previous: Node,
    current: Node,
    last_contact_side: int,
    config: GrammarConfig | None = None,
) -> float:
    """Return the log score of following ``previous`` with ``current``.

    ``NEGATIVE_INFINITY`` marks a transition the grammar forbids: an unsupported
    type pair, a second bounce before the next contact, or two events closer
    together than the minimum flight duration.
    """

    config = (config or DEFAULT_GRAMMAR).for_clip(None)
    score = _transition_prior(previous, current, config)
    if score <= NEGATIVE_INFINITY / 2:
        return NEGATIVE_INFINITY
    if (
        current.event_type == "contact"
        and last_contact_side
        and current.side
        and current.side == last_contact_side
    ):
        score += config.side_violation_penalty
    return score


def path_score(
    nodes: list[Node], path: list[int] | tuple[int, ...], config: GrammarConfig | None = None
) -> float:
    """Return the honest grammar score of one explicit lattice path."""

    resolved = (config or DEFAULT_GRAMMAR).for_clip(None)
    if not path:
        return 0.0
    first = nodes[path[0]]
    score = _start_prior(first, resolved) + first.gain
    carried = _contact_side(first, 0)
    for left, right in zip(path, path[1:], strict=False):
        current = nodes[right]
        transition = _transition_prior(nodes[left], current, resolved)
        if transition <= NEGATIVE_INFINITY / 2:
            return NEGATIVE_INFINITY
        if current.event_type == "contact" and carried and current.side == carried:
            transition += resolved.side_violation_penalty
        score += transition + current.gain
        carried = _contact_side(current, carried)
    score += resolved.end_log_prior[nodes[path[-1]].event_type]
    return float(score)


def _contact_side(node: Node, inherited: int) -> int:
    if node.event_type == "contact" and node.side:
        return node.side
    return inherited


SIDES = (-1, 0, 1)


def viterbi_exact_side(
    nodes: list[Node], config: GrammarConfig
) -> tuple[list[int], np.ndarray, float]:
    """Decode with the carried contact side inside the dynamic-programming state.

    ``viterbi`` keeps one score per node and evaluates the side-alternation
    penalty against the side of that node's *best* partial path, which is a
    greedy approximation: a slightly worse prefix that leaves the ball on the
    other side can extend into a strictly better complete path, and on this
    cohort it does on 5 of 191 points even with the shipped net line.  The side
    takes three values, so making it part of the state is exact at three times
    the cost.
    """

    count = len(nodes)
    if not count:
        return [], np.zeros(0, dtype=np.float64), 0.0
    successors, predecessors, penalty = _transition_tables(nodes, config)
    index_of = {side: position for position, side in enumerate(SIDES)}
    best = np.full((count, len(SIDES)), NEGATIVE_INFINITY)
    parent = np.full((count, len(SIDES)), -1, dtype=np.int64)
    parent_side = np.full((count, len(SIDES)), 0, dtype=np.int64)
    alpha = np.full((count, len(SIDES)), NEGATIVE_INFINITY)
    for position, node in enumerate(nodes):
        start_side = index_of[_contact_side(node, 0)]
        start = _start_prior(node, config) + node.gain
        best[position, start_side] = start
        alpha[position, start_side] = start
        incoming: dict[int, list[float]] = {start_side: [start]}
        for prior, transition in predecessors[position]:
            for carried in SIDES:
                source = index_of[carried]
                if best[prior, source] <= NEGATIVE_INFINITY / 2:
                    continue
                score = transition
                if carried and node.side and node.side == carried:
                    score += penalty[position]
                target = index_of[_contact_side(node, carried)]
                candidate = best[prior, source] + score + node.gain
                if candidate > best[position, target]:
                    best[position, target] = candidate
                    parent[position, target] = prior
                    parent_side[position, target] = carried
                incoming.setdefault(target, []).append(alpha[prior, source] + score + node.gain)
        for target, values in incoming.items():
            alpha[position, target] = _logsumexp(values)

    beta = np.zeros((count, len(SIDES)))
    for position in range(count - 1, -1, -1):
        for carried in SIDES:
            source = index_of[carried]
            if alpha[position, source] <= NEGATIVE_INFINITY / 2:
                continue
            outgoing = [float(config.end_log_prior[nodes[position].event_type])]
            for later, transition in successors[position]:
                score = transition
                if carried and nodes[later].side and nodes[later].side == carried:
                    score += penalty[later]
                target = index_of[_contact_side(nodes[later], carried)]
                outgoing.append(score + nodes[later].gain + beta[later, target])
            beta[position, source] = _logsumexp(outgoing)

    completed = [
        float(alpha[position, side_index]) + float(config.end_log_prior[node.event_type])
        for position, node in enumerate(nodes)
        for side_index in range(len(SIDES))
        if alpha[position, side_index] > NEGATIVE_INFINITY / 2
    ]
    partition = _logsumexp([0.0, *completed])
    joint = np.where(
        alpha > NEGATIVE_INFINITY / 2, np.clip(alpha + beta - partition, -700.0, 0.0), -np.inf
    )
    marginals = np.exp(joint).sum(axis=1)
    marginals = np.clip(marginals, 0.0, 1.0)

    completed_best = best + np.asarray(
        [[config.end_log_prior[node.event_type]] * len(SIDES) for node in nodes]
    )
    flat = int(np.argmax(completed_best))
    terminal, terminal_side = divmod(flat, len(SIDES))
    score = float(completed_best[terminal, terminal_side])
    if score <= 0.0:
        return [], marginals, 0.0
    path = []
    cursor, cursor_side = terminal, terminal_side
    while cursor >= 0:
        path.append(cursor)
        following = int(parent[cursor, cursor_side])
        cursor_side = index_of[int(parent_side[cursor, cursor_side])]
        cursor = following
    path.reverse()
    return path, marginals, score


def viterbi(
    nodes: list[Node], config: GrammarConfig | None = None
) -> tuple[list[int], np.ndarray, float]:
    """Return the best path, the node posterior marginals, and the path score.

    The side-alternation constraint depends on the side of the last contact, so
    the dynamic program carries it forward on the best partial path; the
    sum-product pass uses the same carried side, which keeps both passes on one
    lattice.
    """

    config = (config or DEFAULT_GRAMMAR).for_clip(None)
    count = len(nodes)
    if not count:
        return [], np.zeros(0, dtype=np.float64), 0.0
    best = np.full(count, NEGATIVE_INFINITY)
    parent = np.full(count, -1, dtype=np.int64)
    side = np.zeros(count, dtype=np.int64)
    alpha = np.full(count, NEGATIVE_INFINITY)
    for position, node in enumerate(nodes):
        start = _start_prior(node, config) + node.gain
        best[position] = start
        alpha[position] = start
        side[position] = _contact_side(node, 0)
        incoming = [start]
        for prior in range(position):
            if nodes[prior].frame >= node.frame:
                continue
            score = _transition_prior(nodes[prior], node, config)
            if score <= NEGATIVE_INFINITY / 2:
                continue
            if node.event_type == "contact" and int(side[prior]) and node.side == int(side[prior]):
                score += config.side_violation_penalty
            candidate = best[prior] + score + node.gain
            if candidate > best[position]:
                best[position] = candidate
                parent[position] = prior
                side[position] = _contact_side(node, int(side[prior]))
            incoming.append(alpha[prior] + score + node.gain)
        alpha[position] = _logsumexp(incoming)

    beta = np.zeros(count)
    for position in range(count - 1, -1, -1):
        outgoing = [float(config.end_log_prior[nodes[position].event_type])]
        for later in range(position + 1, count):
            if nodes[later].frame <= nodes[position].frame:
                continue
            score = _transition_prior(nodes[position], nodes[later], config)
            if score <= NEGATIVE_INFINITY / 2:
                continue
            if (
                nodes[later].event_type == "contact"
                and int(side[position])
                and nodes[later].side == int(side[position])
            ):
                score += config.side_violation_penalty
            outgoing.append(score + nodes[later].gain + beta[later])
        beta[position] = _logsumexp(outgoing)

    # Every path ends at exactly one node, so the partition sums ``alpha`` over
    # the last node plus the empty path; ``beta`` already carries termination.
    partition = _logsumexp(
        [
            0.0,
            *(
                alpha[position] + config.end_log_prior[node.event_type]
                for position, node in enumerate(nodes)
            ),
        ]
    )
    marginals = np.exp(np.clip(alpha + beta - partition, -700.0, 0.0))

    completed_best = np.asarray(
        [
            best[position] + config.end_log_prior[node.event_type]
            for position, node in enumerate(nodes)
        ]
    )
    terminal = int(np.argmax(completed_best))
    score = float(completed_best[terminal])
    if score <= 0.0:
        return [], marginals, 0.0
    path = []
    cursor = terminal
    while cursor >= 0:
        path.append(cursor)
        cursor = int(parent[cursor])
    path.reverse()
    return path, marginals, score


def _transition_tables(nodes: list[Node], config: GrammarConfig) -> tuple[list, list, list]:
    """Precompute the side-independent half of every legal transition.

    ``transition_score`` is the decoder's hot loop and its only state-dependent
    term is the side penalty, which applies exactly when the arriving node is a
    contact whose side equals the side carried on the partial path.  Splitting
    the two halves lets the k-best and analysis passes run the same arithmetic as
    ``viterbi`` without re-resolving the config per edge.
    """

    count = len(nodes)
    successors: list[list[tuple[int, float]]] = [[] for _ in range(count)]
    predecessors: list[list[tuple[int, float]]] = [[] for _ in range(count)]
    penalty = [
        config.side_violation_penalty if node.event_type == "contact" and node.side else 0.0
        for node in nodes
    ]
    for left in range(count):
        for right in range(left + 1, count):
            if nodes[right].frame <= nodes[left].frame:
                continue
            prior = _transition_prior(nodes[left], nodes[right], config)
            if prior <= NEGATIVE_INFINITY / 2:
                continue
            successors[left].append((right, prior))
            predecessors[right].append((left, prior))
    return successors, predecessors, penalty


def viterbi_tables(nodes: list[Node], config: GrammarConfig | None = None) -> dict:
    """Return the forward/backward max-product tables of one lattice.

    ``forward[i]`` is the best score of a partial path ending at ``i`` (the same
    array ``viterbi`` maximises over) and ``backward[i]`` the best score of the
    suffix that follows ``i``, so ``forward[i] + backward[i]`` is the score of the
    best complete path *through* ``i``.  Its shortfall against ``max(forward)`` is
    what the argmax path pays to keep ``i`` off itself, which is the number the
    grammar-term attribution in ``cv.validation.grammar_path_arms`` reads.
    """

    config = (config or DEFAULT_GRAMMAR).for_clip(None)
    count = len(nodes)
    if not count:
        return {
            "forward": np.zeros(0),
            "parent": np.zeros(0, dtype=np.int64),
            "side": np.zeros(0, dtype=np.int64),
            "backward": np.zeros(0),
            "path": [],
        }
    successors, predecessors, penalty = _transition_tables(nodes, config)
    forward = np.full(count, NEGATIVE_INFINITY)
    parent = np.full(count, -1, dtype=np.int64)
    side = np.zeros(count, dtype=np.int64)
    for position, node in enumerate(nodes):
        forward[position] = _start_prior(node, config) + node.gain
        side[position] = _contact_side(node, 0)
        for prior, transition in predecessors[position]:
            score = transition
            if int(side[prior]) and node.side and node.side == int(side[prior]):
                score += penalty[position]
            candidate = forward[prior] + score + node.gain
            if candidate > forward[position]:
                forward[position] = candidate
                parent[position] = prior
                side[position] = _contact_side(node, int(side[prior]))
    backward = np.asarray([config.end_log_prior[node.event_type] for node in nodes], dtype=float)
    for position in range(count - 1, -1, -1):
        for later, transition in successors[position]:
            score = transition
            if (
                int(side[position])
                and nodes[later].side
                and nodes[later].side == int(side[position])
            ):
                score += penalty[later]
            value = score + nodes[later].gain + backward[later]
            if value > backward[position]:
                backward[position] = value
    completed = np.asarray(
        [
            forward[position] + config.end_log_prior[node.event_type]
            for position, node in enumerate(nodes)
        ]
    )
    terminal = int(np.argmax(completed))
    path: list[int] = []
    if completed[terminal] > 0.0:
        cursor = terminal
        while cursor >= 0:
            path.append(cursor)
            cursor = int(parent[cursor])
        path.reverse()
    return {
        "forward": forward,
        "parent": parent,
        "side": side,
        "backward": backward,
        "path": path,
    }


def nbest_paths(
    nodes: list[Node], config: GrammarConfig | None = None, count: int = NBEST_PATHS
) -> list[tuple[float, list[int]]]:
    """Return up to ``count`` highest-scoring distinct paths, best first.

    List Viterbi: every node keeps its ``count`` best partial paths, each with
    the contact side it carries, so the side-alternation penalty is evaluated per
    partial path exactly as the single-best pass evaluates it.
    """

    config = (config or DEFAULT_GRAMMAR).for_clip(None)
    if not nodes or count < 1:
        return []
    successors, predecessors, penalty = _transition_tables(nodes, config)
    del successors
    # entries[i][rank] = (score, parent index, parent rank, carried side)
    entries: list[list[tuple[float, int, int, int]]] = []
    for position, node in enumerate(nodes):
        candidates = [
            (
                _start_prior(node, config) + node.gain,
                -1,
                -1,
                _contact_side(node, 0),
            )
        ]
        for prior, transition in predecessors[position]:
            for rank, (partial, _p, _r, carried) in enumerate(entries[prior]):
                score = transition
                if carried and node.side and node.side == carried:
                    score += penalty[position]
                candidates.append(
                    (partial + score + node.gain, prior, rank, _contact_side(node, carried))
                )
        candidates.sort(key=lambda item: -item[0])
        entries.append(candidates[:count])
    ends = [
        (
            entries[position][rank][0] + config.end_log_prior[nodes[position].event_type],
            position,
            rank,
        )
        for position in range(len(nodes))
        for rank in range(len(entries[position]))
    ]
    ends.sort(key=lambda item: (-item[0], item[1], item[2]))
    output: list[tuple[float, list[int]]] = []
    seen: set[tuple[int, ...]] = set()
    for score, position, rank in ends:
        if score <= 0.0 or len(output) >= count:
            break
        path: list[int] = []
        cursor, cursor_rank = position, rank
        while cursor >= 0:
            path.append(cursor)
            _s, cursor, cursor_rank = (
                entries[cursor][cursor_rank][0],
                entries[cursor][cursor_rank][1],
                entries[cursor][cursor_rank][2],
            )
        path.reverse()
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        output.append((float(score), path))
    return output


@dataclass(frozen=True)
class DecodedEvent:
    clip: str
    frame: int
    event_type: str
    probability: float
    marginal: float
    row: int
    on_best_path: bool
    path_vote: float = 1.0
    terminal_kind: str | None = None
    terminal_marginal: float = 0.0
    preceding_bounce_frame: int | None = None
    preceding_bounce_row: int | None = None
    terminal_support_marginal: float | None = None
    acceptance_marginal: float | None = None


def decode_clip(
    probabilities: np.ndarray,
    frames: np.ndarray,
    court_y: np.ndarray,
    rows: np.ndarray,
    clip: str,
    *,
    config: GrammarConfig | None = None,
    emission_mode: str = BEST_PATH,
    off_path_floor: float = OFF_PATH_MARGINAL_FLOOR,
    nbest: int = NBEST_PATHS,
    evidence_log_prior: np.ndarray | None = None,
    evidence_admits: np.ndarray | None = None,
) -> list[DecodedEvent]:
    """Decode one point into emitted events with their marginals.

    ``emission_mode`` decides which lattice nodes leave the decoder.  The
    shipped ``best_path`` emits the Viterbi argmax only.  ``path_marginal`` adds
    every node whose sum-product posterior clears ``off_path_floor``, so a node
    the argmax path skips is still offered to the abstention threshold on its own
    posterior.  ``nbest_union`` adds every node on one of the ``nbest`` best
    paths and scores it by the fraction of those paths that contain it, which is
    the discrete counterpart of the marginal.

    In every mode the score the abstention threshold reads is ``marginal``; only
    ``nbest_union`` replaces it with the path vote.
    """

    if emission_mode not in EMISSION_MODES:
        raise ValueError(f"unknown emission mode: {emission_mode}")
    resolved = (config or DEFAULT_GRAMMAR).for_clip(clip)

    # The second-bounce suffix supplies useful topology evidence, but it must
    # not manufacture confidence for its preceding bounce.  Compute that
    # predecessor's marginal in the identical lattice without the suffix.  The
    # emission layer applies its normal operating threshold to this value when
    # deciding whether the support bounce may leave the decoder.
    support_marginals: dict[tuple[int, str], float] = {}
    ordinary_marginals_by_key: dict[tuple[int, str], float] = {}
    ordinary_nodes: list[Node] = []
    ordinary_path: list[int] = []
    if resolved.terminal_second_bounce and not resolved.emit_terminal_support_bounce:
        ordinary = replace(resolved, terminal_second_bounce=False)
        ordinary_nodes = build_nodes(
            probabilities,
            frames,
            court_y,
            config=ordinary,
            clip=clip,
            evidence_log_prior=evidence_log_prior,
            evidence_admits=evidence_admits,
        )
        if ordinary_nodes:
            if ordinary.exact_side_state:
                ordinary_path, ordinary_marginals, _ordinary_score = viterbi_exact_side(
                    ordinary_nodes, ordinary
                )
            else:
                ordinary_path, ordinary_marginals, _ordinary_score = viterbi(
                    ordinary_nodes, ordinary
                )
            support_marginals = {
                (node.index, node.event_type): float(ordinary_marginals[position])
                for position, node in enumerate(ordinary_nodes)
            }
            ordinary_marginals_by_key = dict(support_marginals)
    elif resolved.terminal_second_bounce and resolved.terminal_requires_no_ordinary_contact:
        ordinary = replace(resolved, terminal_second_bounce=False)
        ordinary_nodes = build_nodes(
            probabilities,
            frames,
            court_y,
            config=ordinary,
            clip=clip,
            evidence_log_prior=evidence_log_prior,
            evidence_admits=evidence_admits,
        )
        if ordinary_nodes:
            if ordinary.exact_side_state:
                ordinary_path, _ordinary_marginals, _ordinary_score = viterbi_exact_side(
                    ordinary_nodes, ordinary
                )
            else:
                ordinary_path, _ordinary_marginals, _ordinary_score = viterbi(
                    ordinary_nodes, ordinary
                )

    nodes = build_nodes(
        probabilities,
        frames,
        court_y,
        config=resolved,
        clip=clip,
        evidence_log_prior=evidence_log_prior,
        evidence_admits=evidence_admits,
    )
    if resolved.terminal_second_bounce and resolved.terminal_requires_no_ordinary_contact:
        ordinary_contact_frames = [
            ordinary_nodes[position].frame
            for position in ordinary_path
            if ordinary_nodes[position].event_type == "contact"
        ]
        nodes = [
            node
            for node in nodes
            if not node.terminal or not any(frame > node.frame for frame in ordinary_contact_frames)
        ]
    if resolved.exact_side_state:
        path, marginals, _ = viterbi_exact_side(nodes, resolved)
    else:
        path, marginals, _ = viterbi(nodes, resolved)
    chosen = set(path)
    if resolved.terminal_second_bounce:
        # Preserve the ordinary best path exactly, then add only the recovered
        # second-bounce suffix.  A terminal alternative must never displace an
        # already accepted contact/bounce elsewhere in the point.
        node_positions = {
            (node.index, node.event_type, node.terminal): position
            for position, node in enumerate(nodes)
        }
        ordinary_selected = {
            node_positions[(node.index, node.event_type, False)]
            for node in (ordinary_nodes[position] for position in ordinary_path)
            if (node.index, node.event_type, False) in node_positions
        }
        chosen = ordinary_selected | {position for position in path if nodes[position].terminal}
    votes = {position: 1.0 for position in chosen}
    selected = set(chosen)
    if emission_mode == PATH_MARGINAL:
        selected |= {
            position
            for position in range(len(nodes))
            if float(marginals[position]) >= off_path_floor
        }
    elif emission_mode == NBEST_UNION:
        paths = nbest_paths(nodes, resolved, nbest)
        counted: dict[int, int] = defaultdict(int)
        for _score, candidate in paths:
            for position in set(candidate):
                counted[position] += 1
        total = max(len(paths), 1)
        votes = {position: value / total for position, value in counted.items()}
        selected |= set(counted)
    # Bounce/terminal-bounce are mutually exclusive states of ONE observation.
    # Sum their posterior mass so adding the terminal state does not split the
    # physical bounce's confidence or emit its pixels twice.
    groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for position, node in enumerate(nodes):
        groups[(node.index, node.event_type)].append(position)
    emitted_groups = set()
    terminal_predecessors: dict[tuple[int, str], tuple[int, int]] = {}
    for path_position, member in enumerate(path):
        if path_position == 0 or not nodes[member].terminal:
            continue
        predecessor = nodes[path[path_position - 1]]
        terminal_predecessors[(predecessor.index, predecessor.event_type)] = (
            predecessor.frame,
            predecessor.index,
        )
    output = []
    for position in sorted(selected):
        node = nodes[position]
        key = (node.index, node.event_type)
        if key in emitted_groups:
            continue
        emitted_groups.add(key)
        members = groups[key]
        class_index = CLASSES.index(node.event_type)
        vote = min(1.0, sum(float(votes.get(member, 0.0)) for member in members))
        marginal = (
            vote
            if emission_mode == NBEST_UNION
            else min(1.0, sum(float(marginals[member]) for member in members))
        )
        terminal_members = [
            member for member in members if nodes[member].terminal and member in selected
        ]
        terminal_marginal = sum(
            float(votes.get(member, 0.0))
            if emission_mode == NBEST_UNION
            else float(marginals[member])
            for member in terminal_members
        )
        preceding_frame = None
        preceding_row = None
        for member in terminal_members:
            if member in chosen:
                preceding = nodes[path[path.index(member) - 1]]
                preceding_frame = preceding.frame
                preceding_row = int(rows[preceding.index])
        support = terminal_predecessors.get(key)
        acceptance_marginal = (
            None
            if terminal_members
            else ordinary_marginals_by_key.get(key)
            if resolved.terminal_second_bounce
            else None
        )
        output.append(
            DecodedEvent(
                clip=clip,
                frame=node.frame,
                event_type=node.event_type,
                probability=float(probabilities[node.index, class_index]),
                marginal=marginal,
                row=int(rows[node.index]),
                on_best_path=any(member in chosen for member in members),
                path_vote=vote,
                terminal_kind="second_bounce" if terminal_members else None,
                terminal_marginal=terminal_marginal,
                preceding_bounce_frame=preceding_frame,
                preceding_bounce_row=preceding_row,
                terminal_support_marginal=(
                    support_marginals.get(key) if support is not None else None
                ),
                acceptance_marginal=acceptance_marginal,
            )
        )
    return output


def decode(
    probabilities: np.ndarray,
    clips: np.ndarray,
    frames: np.ndarray,
    court_y: np.ndarray,
    *,
    config: GrammarConfig | None = None,
    emission_mode: str = BEST_PATH,
    off_path_floor: float = OFF_PATH_MARGINAL_FLOOR,
    nbest: int = NBEST_PATHS,
    evidence_log_prior: np.ndarray | None = None,
    evidence_admits: np.ndarray | None = None,
) -> list[DecodedEvent]:
    """Decode every point of a cohort."""

    output: list[DecodedEvent] = []
    clips = np.asarray(clips)
    for clip in sorted(set(clips.tolist())):
        mask = np.flatnonzero(clips == clip)
        output.extend(
            decode_clip(
                probabilities[mask],
                frames[mask],
                court_y[mask],
                mask,
                str(clip),
                config=config,
                emission_mode=emission_mode,
                off_path_floor=off_path_floor,
                nbest=nbest,
                evidence_log_prior=(
                    None if evidence_log_prior is None else evidence_log_prior[mask]
                ),
                evidence_admits=None if evidence_admits is None else evidence_admits[mask],
            )
        )
    return output


POINT_GRAMMAR_ABSENT = {
    "schema": "event_grammar_decoder_no_point_grammar_v1",
    "present": False,
    "reason": (
        "this path decodes the point grammar itself with a linear-chain Viterbi and "
        "takes its location from the model's x/y head, so cv.pipeline.point_grammar "
        "never runs and its proposal-derived fields are absent"
    ),
    "absent_fields": [
        "point_grammar.in_play",
        "point_grammar.chain_evidence",
        "point_grammar.structure_span_id",
        "point_grammar.confidence",
        "point_gate_verdict",
        "point_gate_failure_reasons",
        "gate_held",
        "production_scope",
        "location.court_x_m",
        "location.court_y_m",
        "location.proposal_offset_frames",
    ],
    "replacements": {
        "in_play": "membership of the decoder's best path; every emitted row is on it",
        "chain_evidence": "path_marginal and abstention_gap",
        "location": "location.source == event_video_model_xy_head",
    },
}


def emission_rows(
    events,
    store,
    probabilities: np.ndarray,
    predicted_offset: np.ndarray,
    predicted_xy: np.ndarray,
    centres: np.ndarray,
    fps: dict[str, float],
    marginal_threshold: float,
    *,
    time_offset: np.ndarray | None = None,
    time_distribution: tuple[np.ndarray, np.ndarray] | None = None,
    correct_frames: bool = False,
) -> list[dict]:
    """Turn decoded best-path events into ``event_model_v3`` emission rows.

    ``store`` is any object exposing ``clips()``, ``frames()`` and
    ``broadcasts()`` over the manifest rows the probabilities were computed on.

    With ``correct_frames`` the emitted integer frame moves by the time head's
    expected offset, clipped to ``MAX_TIME_CORRECTION``; two corrected rows that
    land on the same (clip, type, frame) collapse to the one with the larger
    marginal.  Without it the row's own frame is emitted, which is what the first
    pass did.

    ``time_distribution`` is the optional ``(pmf, offset grid)`` pair from the
    prediction artifact.  When it is given, each row carries the model's whole
    local timing softmax as additive evidence, plus the ``time_neighborhood``
    block holding the same evidence for the available rows of its own clip
    within ``neighborhood.RADIUS_NATIVE_FRAMES`` of its candidate frame.  The
    emitted frame, the mean time and every decision above stay exactly as they
    were without it.
    """

    clips = store.clips()
    frames = store.frames()
    broadcasts = store.broadcasts()
    court_x = (
        store.court_x()
        if hasattr(store, "court_x")
        else np.full(len(frames), math.nan, dtype=np.float32)
    )
    court_y = (
        store.court_y()
        if hasattr(store, "court_y")
        else np.full(len(frames), math.nan, dtype=np.float32)
    )
    observed = (
        store.track_observed()
        if hasattr(store, "track_observed")
        else np.ones(len(frames), dtype=bool)
    )
    transport_records = (
        store.court_transport_records() if hasattr(store, "court_transport_records") else None
    )
    if transport_records is not None and len(transport_records) != len(frames):
        raise ValueError("court transport provenance must align with proposal rows")
    # An unobserved row alone is not an exit (it can be an occlusion).  A track
    # exit needs an observed final location at a native image edge followed by
    # a missing observation on the next available frame of the same clip.
    exit_after = np.zeros(len(frames), dtype=bool)
    court_exit_after = np.zeros(len(frames), dtype=bool)
    for clip in sorted(set(clips.tolist())):
        positions = np.flatnonzero(clips == clip)
        for position, current in enumerate(positions[:-1]):
            later = positions[position + 1]
            if int(frames[later]) <= int(frames[current]):
                continue
            x, y = centres[current]
            at_edge = (
                min(float(x), NATIVE_WIDTH - float(x), float(y), NATIVE_HEIGHT - float(y))
                <= TRACK_EXIT_EDGE_PX
            )
            exit_after[current] = bool(observed[current] and not observed[later] and at_edge)
            # Retain the later court-plane projection as a diagnostic. An
            # airborne rebound can project outside the court even after a valid
            # in-court bounce; this does not determine where that bounce landed
            # and must not certify a physical out ending.
            following = positions[position + 1 : position + 13]
            court_exit_after[current] = bool(
                observed[current]
                and any(
                    observed[other]
                    and math.isfinite(float(court_x[other]))
                    and math.isfinite(float(court_y[other]))
                    and (
                        float(court_x[other]) < 0.0
                        or float(court_x[other]) > 1.0
                        or float(court_y[other]) < 0.0
                        or float(court_y[other]) > 1.0
                    )
                    for other in following
                )
            )
    distribution = (
        None
        if time_distribution is None
        else timing.validated(*time_distribution, rows=len(frames))
    )
    # One sorted per-clip frame index for the whole pass, so each row's timing
    # neighborhood is a binary search instead of a scan over every source row.
    neighborhood_index = (
        None
        if distribution is None
        else neighborhood.build_index(clips, frames, allow_ambiguous=True)
    )
    rows = []
    for event in events:
        row = event.row
        try:
            rate = float(fps[str(broadcasts[row])])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("missing or invalid event frame rate") from error
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("event frame rate must be positive and finite")
        if time_offset is not None:
            shift = float(np.clip(time_offset[row], -MAX_TIME_CORRECTION, MAX_TIME_CORRECTION))
        else:
            shift = float(np.clip(predicted_offset[row], -1.0, 1.0))
        subframe = float(frames[row]) + shift
        emitted = float(round(subframe)) if correct_frames else float(frames[row])
        native_x = float(centres[row, 0] + predicted_xy[row, 0] * CROP_HALF_WIDTH)
        native_y = float(centres[row, 1] + predicted_xy[row, 1] * CROP_HALF_HEIGHT)
        acceptance_marginal = float(
            event.acceptance_marginal
            if event.acceptance_marginal is not None
            else event.terminal_support_marginal
            if event.terminal_support_marginal is not None
            else event.marginal
        )
        rows.append(
            {
                "clip": str(clips[row]),
                "match_id": str(broadcasts[row]),
                "event_type": event.event_type,
                "frame": emitted,
                "candidate_frame": float(frames[row]),
                "confidence": float(event.marginal),
                "probability": float(event.probability),
                "class_probabilities": {
                    name: float(probabilities[row, index]) for index, name in enumerate(CLASSES)
                },
                "abstain": bool(acceptance_marginal < marginal_threshold),
                "decision_threshold": float(marginal_threshold),
                "abstention_gap": float(1.0 - event.marginal),
                "path_marginal": float(event.marginal),
                # This normally equals ``path_marginal``.  A second-bounce
                # support state is the one exception: its full-lattice
                # posterior includes the terminal suffix it is witnessing, so
                # its acceptance score is the counterfactual ordinary-grammar
                # marginal instead.
                "acceptance_marginal": acceptance_marginal,
                "on_best_path": bool(event.on_best_path),
                "path_vote": float(event.path_vote),
                "point_grammar": dict(POINT_GRAMMAR_ABSENT),
                "location": {
                    "source": "event_video_model_xy_head",
                    "image_x": native_x,
                    "image_y": native_y,
                    "image_coordinate_space": "native_1920x1080",
                    "frame_subpixel": subframe,
                    "time_seconds": (subframe - 1.0) / rate,
                    "time_reference": "clip_start",
                    "frame_index_origin": 1,
                    "fps": rate,
                    "court_x_m": None,
                    "court_y_m": None,
                    "court_x_fraction": (
                        float(court_x[row]) if math.isfinite(float(court_x[row])) else None
                    ),
                    "court_y_fraction": (
                        float(court_y[row]) if math.isfinite(float(court_y[row])) else None
                    ),
                    "track_observed": bool(observed[row]),
                },
            }
        )
        if distribution is not None:
            # Offsets are relative to this row's own native crop frame, never to
            # the emitted or corrected epoch, and the scores are uncalibrated.
            rows[-1]["time_distribution"] = timing.record(
                distribution[0][row], distribution[1][row]
            )
            # The neighbouring crop rows scored this same moment from their own
            # windows; they are the separate evidence for a timing, anchored on
            # the row's own candidate frame rather than the emitted one.
            if neighborhood_index[str(clips[row])] is None:
                rows[-1]["time_neighborhood_status"] = "unavailable_ambiguous_native_candidate_rows"
            else:
                rows[-1][neighborhood.KEY] = neighborhood.record(
                    row,
                    index=neighborhood_index,
                    clip=str(clips[row]),
                    candidate_frame=float(frames[row]),
                    probabilities=probabilities,
                    classes=CLASSES,
                    pmf=distribution[0],
                    grid=distribution[1],
                )
        if transport_records is not None and transport_records[row] is not None:
            # Court fractions still describe the native proposal pixel/epoch,
            # not the separately predicted XY/time head or airborne ball XYZ.
            rows[-1]["location"]["court_transport"] = dict(transport_records[row])
        if event.terminal_support_marginal is not None:
            rows[-1]["terminal_support"] = {
                "status": "withheld" if rows[-1]["abstain"] else "independently_accepted",
                "source": "event_grammar_decoder_second_bounce_support",
                "counterfactual_nonterminal_marginal": float(event.terminal_support_marginal),
                "reason": (
                    "used as a second-bounce grammar witness; emission requires its "
                    "nonterminal decoder marginal to clear the operating threshold"
                ),
            }
        if exit_after[row]:
            rows[-1]["track_exit_after"] = True
        if court_exit_after[row]:
            rows[-1]["court_exit_after"] = True
        if event.terminal_kind is not None:
            preceding_in_bounds = None
            if event.preceding_bounce_row is not None:
                preceding_x = float(court_x[event.preceding_bounce_row])
                preceding_y = float(court_y[event.preceding_bounce_row])
                if math.isfinite(preceding_x) and math.isfinite(preceding_y):
                    low, high = -COURT_BOUNDARY_TOLERANCE, 1.0 + COURT_BOUNDARY_TOLERANCE
                    preceding_in_bounds = low <= preceding_x <= high and low <= preceding_y <= high
            rows[-1]["terminal_evidence"] = {
                "status": "candidate",
                "source": "event_grammar_decoder_terminal_state",
                "termination_kind": event.terminal_kind,
                "marginal": event.terminal_marginal,
                "preceding_bounce_candidate_frame": event.preceding_bounce_frame,
                "preceding_bounce_in_bounds": preceding_in_bounds,
                "reason": "requires first-bounce court/serve validity and terminal visual validation",
            }
    if correct_frames:
        best: dict[tuple[str, str, float], dict] = {}
        for row in rows:
            key = (row["clip"], row["event_type"], row["frame"])
            if key not in best or row["path_marginal"] > best[key]["path_marginal"]:
                best[key] = row
        rows = list(best.values())
    return sorted(rows, key=lambda row: (row["clip"], row["frame"], row["event_type"]))


def point_end_rows(
    rows: list[dict], *, point_contexts: Mapping[str, Mapping] | None = None
) -> list[dict]:
    """Emit endings only when automatic terminal evidence determines their kind.

    The final lattice member is not itself a point ending: a contact may be
    followed by an unseen flight, and a net hit may be followed by the terminal
    ground impact.  This derives the tennis ending rule from decoded physical
    evidence instead: a first out bounce, a visible second bounce after an in
    bounce, a net hit with no ensuing ground impact, or an observed track exit.
    """

    by_clip: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["abstain"] is not True and row.get("event_type") in EVENT_TYPES:
            by_clip[row["clip"]].append(row)
    output = []
    for clip, members in sorted(by_clip.items()):
        context = None if point_contexts is None else point_contexts.get(clip)
        ending = _decoded_ending(members, point_context=context)
        if ending is None:
            continue
        terminal, evidence = ending
        output.append(
            {
                "clip": clip,
                "match_id": terminal["match_id"],
                "event_type": "point_end",
                "frame": float(terminal["frame"]),
                "confidence": terminal["confidence"],
                "probability": terminal["probability"],
                "abstain": False,
                "location": dict(terminal["location"]),
                "point_end": {
                    "source": "event_grammar_decoder_ending_kind",
                    "termination_kind": evidence["termination_kind"],
                    "evidence": dict(evidence),
                    "chain_length": len(members),
                    "terminal_event_type": terminal["event_type"],
                },
            }
        )
    return output


def _court_in_bounds(row: dict) -> bool | None:
    """Return whether an event's automatic court-plane location is in court."""

    location = row.get("location") or {}
    try:
        x = float(location["court_x_fraction"])
        y = float(location["court_y_fraction"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    low, high = -COURT_BOUNDARY_TOLERANCE, 1.0 + COURT_BOUNDARY_TOLERANCE
    return low <= x <= high and low <= y <= high


def _decoded_ending(
    members: list[dict], *, point_context: Mapping | None = None
) -> tuple[dict, dict] | None:
    """Find one fail-closed point ending in an accepted decoded path."""

    members = sorted(members, key=lambda row: float(row["frame"]))
    if not members or any(row.get("gate_held") is True for row in members):
        return None
    bounces = [row for row in members if row["event_type"] == "bounce"]
    last = members[-1]
    if last.get("on_best_path") is False:
        return None

    candidates: list[tuple[dict, dict]] = []
    terminal = last.get("terminal_evidence") or {}
    if (
        last["event_type"] == "bounce"
        and terminal.get("termination_kind") == "second_bounce"
        and terminal.get("preceding_bounce_candidate_frame") is not None
    ):
        preceding = next(
            (
                row
                for row in reversed(bounces[:-1])
                if float(row["frame"]) == float(terminal["preceding_bounce_candidate_frame"])
            ),
            None,
        )
        support_in_bounds = (
            _court_in_bounds(preceding)
            if preceding is not None
            else terminal.get("preceding_bounce_in_bounds")
        )
        if support_in_bounds is True:
            candidates.append(
                (
                    last,
                    {
                        "status": "confirmed",
                        "source": "event_grammar_decoder_terminal_state",
                        "termination_kind": "second_bounce",
                        "first_bounce_frame": float(terminal["preceding_bounce_candidate_frame"]),
                        "first_bounce_in_bounds": True,
                        "first_bounce_emitted": preceding is not None,
                    },
                )
            )

    # A failed net return can reach the court after the cord impact.  Its
    # ground impact, not the earlier net hit, is the owner-defined ending.
    if last["event_type"] == "bounce":
        latest_net = max(
            (row for row in members if row["event_type"] == "net_hit"),
            key=lambda row: float(row["frame"]),
            default=None,
        )
        later_contact = (
            any(
                row["event_type"] == "contact" and float(row["frame"]) > float(latest_net["frame"])
                for row in members
            )
            if latest_net is not None
            else False
        )
        if latest_net is not None and not later_contact:
            candidates.append(
                (
                    last,
                    {
                        "status": "confirmed",
                        "source": "event_grammar_decoder_net_ground",
                        "termination_kind": "ground_after_net_hit",
                        "net_hit_frame": float(latest_net["frame"]),
                    },
                )
            )

    # A bounce outside the doubles court is terminal only when it is the first
    # bounce after its originating contact.  A winner's *second* bounce can be
    # beyond the baseline, so it must be handled by the terminal-state rule
    # above rather than mistaken for an out first bounce.
    for bounce in bounces:
        preceding_contact = max(
            (
                row
                for row in members
                if row["event_type"] == "contact" and float(row["frame"]) < float(bounce["frame"])
            ),
            key=lambda row: float(row["frame"]),
            default=None,
        )
        following_contact = min(
            (
                row
                for row in members
                if row["event_type"] == "contact" and float(row["frame"]) > float(bounce["frame"])
            ),
            key=lambda row: float(row["frame"]),
            default=None,
        )
        first_since_contact = preceding_contact is not None and not any(
            float(preceding_contact["frame"]) < float(other["frame"]) < float(bounce["frame"])
            for other in bounces
        )
        reset_or_end = following_contact is None or (
            float(following_contact["frame"]) - float(bounce["frame"]) > ATTEMPT_RESET_GAP_FRAMES
        )
        if first_since_contact and reset_or_end and _court_in_bounds(bounce) is False:
            candidates.append(
                (
                    bounce,
                    {
                        "status": "confirmed",
                        "source": "event_grammar_decoder_court_plane",
                        "termination_kind": "first_bounce_out",
                        "court_in_bounds": _court_in_bounds(bounce),
                        "court_exit_after": bool(bounce.get("court_exit_after")),
                    },
                )
            )

    # A failed net return is still allowed to reach the court.  If no accepted
    # ground impact follows it, the net hit is the only defensible terminal
    # evidence; otherwise the ground impact above carries the ending.
    if last["event_type"] == "net_hit":
        candidates.append(
            (
                last,
                {
                    "status": "confirmed",
                    "source": "event_grammar_decoder_net_hit",
                    "termination_kind": "net_hit",
                },
            )
        )
    if last.get("track_exit_after") is True:
        candidates.append(
            (
                last,
                {
                    "status": "confirmed",
                    "source": "event_grammar_decoder_track_exit",
                    "termination_kind": "leaving_view",
                },
            )
        )
    # An event-only impulse certificate does not validate impact location,
    # service legality or terminal visibility. It may restore interior topology
    # but cannot itself supply a terminal row or a second-bounce ground witness.
    identity_only = {
        (float(row["frame"]), row["event_type"])
        for row in members
        if row.get("event_identity_support", {}).get("certifies_physical_ending") is False
        or row.get("impulse_support", {}).get("supported") is True
    }
    candidates = [
        (row, evidence)
        for row, evidence in candidates
        if (float(row["frame"]), row["event_type"]) not in identity_only
        and not any(
            (float(evidence[name]), kind) in identity_only
            for name, kind in (("first_bounce_frame", "bounce"), ("net_hit_frame", "net_hit"))
            if name in evidence
        )
    ]
    baseline = min(candidates, key=lambda item: float(item[0]["frame"]), default=None)
    if baseline is None or point_context is None:
        return baseline
    try:
        witness = point_context_module.ending_witness(point_context, members)
    except (KeyError, TypeError, ValueError):
        return baseline
    if not witness.get("available"):
        return baseline

    def candidate_score(item: tuple[dict, dict]) -> tuple[float, float]:
        terminal_row, evidence = item
        kind = point_context_module.normalize_consumer_ending_kind(evidence["termination_kind"])
        family = {
            "out": "first_bounce_out",
            "second_bounce": "second_bounce",
            "net": "net_hit",
            "fov_exit": "fov_exit",
        }.get(kind)
        likelihoods = witness["ending_likelihoods"]
        best = max(likelihoods.values())
        likelihood = float(likelihoods.get(family, 0.0))
        context_penalty = min(1.25, max(0.0, math.log(max(best, 1e-6) / max(likelihood, 1e-6))))
        physical = math.log(
            max(float(terminal_row.get("confidence", terminal_row.get("probability", 0.0))), 1e-6)
        )
        evidence["point_context_witness"] = {
            **witness,
            "context_family": family,
            "context_likelihood": likelihood,
            "context_log_penalty": context_penalty,
            "physical_event_log_evidence": physical,
            "combined_log_evidence": physical - context_penalty,
            "maximum_context_log_penalty": 1.25,
            "hard_gate": False,
        }
        return physical - context_penalty, -float(terminal_row["frame"])

    return max(candidates, key=candidate_score)


def rethreshold_emission_rows(rows: list[dict], marginal_threshold: float) -> list[dict]:
    """Re-emit a decoded path at a new calibrated abstention threshold.

    The physical rows and their frozen model probabilities remain unchanged.  Existing point-end
    rows are replaced because the terminal accepted path member can change with the threshold.
    """

    if not 0.0 <= marginal_threshold <= 1.0:
        raise ValueError("marginal threshold must be between zero and one")
    physical = []
    for source in rows:
        event_type = source.get("event_type")
        if event_type == "point_end":
            continue
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unexpected decoded event type: {event_type}")
        if source.get("path_marginal") is None:
            raise ValueError("physical event lacks a decoder path marginal")
        row = dict(source)
        marginal = float(row.get("acceptance_marginal", row["path_marginal"]))
        row["abstain"] = marginal < marginal_threshold
        row["decision_threshold"] = float(marginal_threshold)
        physical.append(row)
    return sorted(
        [*physical, *point_end_rows(physical)],
        key=lambda row: (row["clip"], float(row["frame"]), row["event_type"]),
    )


__all__ = [
    "BEST_PATH",
    "CLASSES",
    "DEFAULT_GRAMMAR",
    "GrammarConfig",
    "MAX_TIME_CORRECTION",
    "TYPE_FLOORS",
    "TYPE_LOG_BONUS",
    "CROP_HALF_HEIGHT",
    "CROP_HALF_WIDTH",
    "POINT_GRAMMAR_ABSENT",
    "emission_rows",
    "point_end_rows",
    "path_score",
    "rethreshold_emission_rows",
    "DecodedEvent",
    "EMISSION_MODES",
    "END_LOG_PRIOR",
    "EVENT_TYPES",
    "LEGACY_NET_LINE",
    "MINIMUM_GAP",
    "NBEST_PATHS",
    "NBEST_UNION",
    "NET_LINE",
    "OFF_PATH_MARGINAL_FLOOR",
    "PROPOSAL_ADMISSION_STRENGTH",
    "PROPOSAL_PRIOR_CAP",
    "PROPOSAL_PRIOR_WEIGHT",
    "PROPOSAL_LOCAL_BASELINE_FRAMES",
    "PROPOSAL_SIGMA_REFERENCE",
    "PROPOSAL_STRENGTH_CONFIDENCE",
    "PROPOSAL_STRENGTH_MODES",
    "PROPOSAL_STRENGTH_SIGMA",
    "PATH_MARGINAL",
    "SIDE_DEAD_BAND",
    "SIDES",
    "nbest_paths",
    "proposal_evidence",
    "proposal_strength",
    "viterbi_exact_side",
    "viterbi_tables",
    "Node",
    "TRANSITION_LOG_PRIOR",
    "build_nodes",
    "decode",
    "decode_clip",
    "side_of",
    "transition_score",
    "viterbi",
]
