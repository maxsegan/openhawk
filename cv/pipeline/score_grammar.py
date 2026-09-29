"""Legal-tennis-score decoding over noisy scoreboard reads.

The scoreboard reader sees one crop per second and gets most of them right, but a handful of
misreads per match are enough to invent point transitions, lose a game, or flip the server.
Tennis scoring is a very small grammar, so most misreads are detectable: from any state the
only legal successors are "unchanged" and "one point to one of the two players", and that one
point determines the games, sets and server that follow.

The decoder is a beam search over that grammar.  A hypothesis is the full match state - the
completed sets, the current games, the current points (tiebreak points are integers), and who
is serving.  Each observed second scores every hypothesis by how many scoreboard fields it
explains, and point transitions pay a penalty that is large when the previous point ended only
a moment ago, which is what stops single-frame misreads from being copied into the ledger.

Two observation layers feed the same decoder.

``score_runs_observations`` reads the slotted ``score_runs.csv`` the reader emits: each row
already claims which digit is the current games and which are the completed sets.  That claim
is wrong whenever the board hides a column - a blank 0-0 games box, a seeding number printed
beside the name, a points box the reader mistook for the games box - and a wrong claim is then
scored as hard evidence.

``board_observations`` keeps the reader's raw left-to-right tokens instead and lets the
decoder decide the column layout.  For each hypothesis the observed tokens are aligned against
the tokens that hypothesis predicts, under a handful of layout variants (a leading seed
number, a hidden trailing games column, a points box read as the games box), each with its own
penalty, and the best alignment scores.  A column the reader did not show is then absence of
evidence rather than evidence of absence, which is what the slotted layer got wrong.

Output is one decoded state per observation second, a per-second confidence from an
observation-dropout ensemble, and the derived point boundaries, so Stage 1 can attach
(server, set, games, points) to every attempt it emits and abstain where the decode is not
supported.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

POINT_WORDS = ("0", "15", "30", "40", "AD")
FIELDS = ("g1", "g2", "p1", "p2", "sets1", "sets2")


@dataclass(frozen=True)
class MatchRules:
    """Explicit scoring format, currently limited to tiebreaks at six games all."""

    best_of: int = 5
    tiebreak_target: int = 7
    deciding_tiebreak_target: int = 10

    def __post_init__(self) -> None:
        if self.best_of not in (3, 5):
            raise ValueError("best_of must be 3 or 5")
        if self.tiebreak_target not in (7, 10) or self.deciding_tiebreak_target not in (7, 10):
            raise ValueError("tiebreak targets must be 7 or 10")

    @property
    def sets_to_win(self) -> int:
        return self.best_of // 2 + 1


DEFAULT_RULES = MatchRules()


@dataclass(frozen=True)
class ScoreState:
    """A complete, legal tennis score."""

    completed: tuple[tuple[int, int], ...]
    games: tuple[int, int]
    points: tuple[int, int]
    server: int

    @property
    def in_tiebreak(self) -> bool:
        return self.games == (6, 6)

    def sets_won(self) -> tuple[int, int]:
        first = sum(1 for a, b in self.completed if a > b)
        return first, len(self.completed) - first

    def is_deciding_set(self, rules: MatchRules = DEFAULT_RULES) -> bool:
        return len(self.completed) == rules.best_of - 1

    def tiebreak_target(self, rules: MatchRules = DEFAULT_RULES) -> int:
        if self.is_deciding_set(rules):
            return rules.deciding_tiebreak_target
        return rules.tiebreak_target

    def is_finished(self, rules: MatchRules = DEFAULT_RULES) -> bool:
        return max(self.sets_won()) >= rules.sets_to_win

    def point_text(self, index: int) -> str:
        if self.in_tiebreak:
            return str(self.points[index])
        return POINT_WORDS[self.points[index]]

    def as_read(self) -> dict[str, str]:
        return {
            "g1": str(self.games[0]),
            "g2": str(self.games[1]),
            "p1": self.point_text(0),
            "p2": self.point_text(1),
            "sets1": " ".join(str(a) for a, _ in self.completed),
            "sets2": " ".join(str(b) for _, b in self.completed),
        }

    def row_tokens(self, index: int) -> tuple[str, ...]:
        """The digit columns the board shows for one row: completed sets, then games."""
        return tuple(str(pair[index]) for pair in self.completed) + (str(self.games[index]),)

    def as_columns(self) -> dict[str, str | int]:
        first, second = self.sets_won()
        return {
            "server": self.server,
            "sets_won_1": first,
            "sets_won_2": second,
            "set_number": len(self.completed) + 1,
            "games_1": self.games[0],
            "games_2": self.games[1],
            "points_1": self.point_text(0),
            "points_2": self.point_text(1),
            "completed_sets": ";".join(f"{a}-{b}" for a, b in self.completed),
        }


def _tiebreak_server(state: ScoreState) -> int:
    """Serve changes after the first tiebreak point, then every two points."""
    played = state.points[0] + state.points[1]
    flips = (played + 1) // 2
    return state.server if flips % 2 == 0 else 3 - state.server


def award_point(state: ScoreState, winner: int, rules: MatchRules = DEFAULT_RULES) -> ScoreState:
    """The single legal successor of ``state`` when ``winner`` (0 or 1) takes the point."""
    loser = 1 - winner
    if state.in_tiebreak:
        target = state.tiebreak_target(rules)
        points = list(state.points)
        points[winner] += 1
        if points[winner] >= target and points[winner] - points[loser] >= 2:
            games = [6, 6]
            games[winner] = 7
            return ScoreState(
                completed=state.completed + ((games[0], games[1]),),
                games=(0, 0),
                points=(0, 0),
                server=3 - state.server,
            )
        return ScoreState(
            completed=state.completed,
            games=state.games,
            points=(points[0], points[1]),
            server=state.server,
        )
    points = list(state.points)
    if points[winner] == 4:  # advantage converted
        return _win_game(state, winner)
    if points[winner] == 3 and points[loser] < 3:
        return _win_game(state, winner)
    if points[winner] == 3 and points[loser] == 4:
        return ScoreState(state.completed, state.games, (3, 3), state.server)
    if points[winner] == 3 and points[loser] == 3:
        points[winner] = 4
        return ScoreState(state.completed, state.games, (points[0], points[1]), state.server)
    points[winner] += 1
    return ScoreState(state.completed, state.games, (points[0], points[1]), state.server)


def _win_game(state: ScoreState, winner: int) -> ScoreState:
    games = list(state.games)
    games[winner] += 1
    loser = 1 - winner
    if games[winner] >= 6 and games[winner] - games[loser] >= 2:
        return ScoreState(
            completed=state.completed + ((games[0], games[1]),),
            games=(0, 0),
            points=(0, 0),
            server=3 - state.server,
        )
    return ScoreState(
        completed=state.completed,
        games=(games[0], games[1]),
        points=(0, 0),
        server=3 - state.server,
    )


def successors(state: ScoreState, rules: MatchRules = DEFAULT_RULES) -> list[ScoreState]:
    """The legal next states.  A finished match has none: no point follows match point."""
    if state.is_finished(rules):
        return []
    return [award_point(state, 0, rules), award_point(state, 1, rules)]


def parse_read(row: dict) -> dict[str, str] | None:
    values = {field_name: (row.get(field_name) or "").strip() for field_name in FIELDS}
    if not values["g1"] or not values["g2"]:
        return None
    return values


def legal_completed_set(games: tuple[int, int]) -> bool:
    """A finished set is 6-n with n <= 4, 7-5, or 7-6; anything else is a misread."""
    high, low = max(games), min(games)
    return (high == 6 and low <= 4) or (high == 7 and low in (5, 6))


def legal_running_games(games: tuple[int, int]) -> bool:
    """A set still in progress cannot already be won: 6-4 and 7-anything are finished."""
    high, low = max(games), min(games)
    if high > 6:
        return False
    return not (high == 6 and low <= 4)


def legal_set_line(completed: Sequence[tuple[int, int]], rules: MatchRules = DEFAULT_RULES) -> bool:
    """Every completed set is legal and the match did not go on after it was won."""
    if len(completed) > rules.best_of:
        return False
    first = second = 0
    for index, pair in enumerate(completed):
        if not legal_completed_set(pair):
            return False
        if max(first, second) >= rules.sets_to_win:
            return False  # a set played after the match was already decided
        if pair[0] > pair[1]:
            first += 1
        else:
            second += 1
        del index
    return True


def state_from_read(
    read: dict[str, str], server: int, rules: MatchRules = DEFAULT_RULES
) -> ScoreState | None:
    """Build the state a read describes, or None when the read is not a legal score."""
    try:
        games = (int(read["g1"]), int(read["g2"]))
    except ValueError:
        return None
    first = [int(value) for value in read["sets1"].split()] if read["sets1"] else []
    second = [int(value) for value in read["sets2"].split()] if read["sets2"] else []
    if len(first) != len(second):
        return None
    completed = tuple(zip(first, second))
    if not legal_set_line(completed, rules):
        return None
    if not legal_running_games(games):
        return None
    tiebreak = games == (6, 6)
    points = []
    for key in ("p1", "p2"):
        token = read[key]
        if tiebreak:
            if not token.isdigit():
                return None
            points.append(int(token))
        else:
            if token not in POINT_WORDS:
                return None
            points.append(POINT_WORDS.index(token))
    if not tiebreak and points[0] == 4 and points[1] == 4:
        return None
    state = ScoreState(completed, games, (points[0], points[1]), server)
    if tiebreak and max(points) > state.tiebreak_target(rules) + 6:
        return None
    if state.is_finished(rules):
        return None  # the match is over; nothing is being played on this board
    return state


def emission_score(
    state: ScoreState, read: dict[str, str] | None, serving_row: int | None
) -> float:
    """Score a slotted ``score_runs.csv`` read against a hypothesis."""
    if read is None:
        return 0.0
    predicted = state.as_read()
    score = 0.0
    for field_name, weight in (
        ("g1", 1.0),
        ("g2", 1.0),
        ("p1", 1.2),
        ("p2", 1.2),
        ("sets1", 0.8),
        ("sets2", 0.8),
    ):
        score += weight if predicted[field_name] == read[field_name] else -weight
    if serving_row in (1, 2):
        actual = _tiebreak_server(state) if state.in_tiebreak else state.server
        score += 0.6 if actual == serving_row else -0.6
    return score


# ---------------- board observations: raw reader tokens, layout decided by the grammar ------

PTS_TOKENS = frozenset({"0", "15", "30", "40", "AD"})

SET_COLUMN_WEIGHT = 1.4
GAMES_COLUMN_WEIGHT = 1.2
POINTS_WEIGHT = 1.0
SERVE_WEIGHT = 0.6
NAME_WEIGHT = 0.4
HIDDEN_COLUMN_PENALTY = 0.3
SEED_PREFIX_PENALTY = 0.5
POINTS_AS_GAMES_PENALTY = 0.5
BLANK_POINTS_BONUS = 0.3
SHAPE_FLOOR = -3.0


@dataclass(frozen=True)
class RowRead:
    """One scoreboard row exactly as the reader saw it, left to right, nothing slotted."""

    columns: tuple[str, ...] = ()
    points: str = ""
    name: str = ""


@dataclass(frozen=True)
class BoardRead:
    """One crop: two rows, the serve marker, and how much the reader is to be believed."""

    t: float
    rows: tuple[RowRead, RowRead] | None = None
    serving_row: int | None = None
    weight: float = 1.0
    run: int = 0

    @property
    def usable(self) -> bool:
        return self.rows is not None


def _token(value: object) -> str:
    text = str(value).strip().upper().rstrip(".")
    return "AD" if text in ("A", "AD") else text


def row_read_from_raw(row: dict) -> RowRead | None:
    """A reader row -> its tokens, with the country code and the name kept apart."""
    if not isinstance(row, dict):
        return None
    columns = []
    for value in row.get("values") or []:
        token = _token(value)
        if token == "AD" or (token.isdigit() and len(token) <= 2):
            columns.append(token)
    points = row.get("points")
    points = "" if points in (None, "", "null", "NULL") else _token(points)
    if points and not (points in PTS_TOKENS or (points.isdigit() and int(points) <= 30)):
        points = ""
    name = "".join(ch for ch in str(row.get("name") or "").upper() if ch.isalpha())
    return RowRead(columns=tuple(columns), points=points, name=name)


def board_reads_from_raw(
    records: Sequence[dict],
    *,
    fps: float = 1.0,
    weights: Sequence[float] | None = None,
    points_only_boards: bool = False,
) -> list[BoardRead]:
    """``vlm_reads.jsonl`` records -> one :class:`BoardRead` per sent crop."""
    reads = []
    for index, record in enumerate(records):
        raw = record.get("raw")
        moment = float(record.get("i", index)) / fps
        weight = 1.0 if weights is None else float(weights[index])
        if not raw or not raw.get("present"):
            reads.append(BoardRead(t=moment, rows=None, weight=weight, run=index))
            continue
        rows = raw.get("rows") or []
        parsed = [row_read_from_raw(row) for row in rows[:2]]
        if len(parsed) != 2 or any(
            row is None or not (row.columns or (points_only_boards and row.points))
            for row in parsed
        ):
            reads.append(BoardRead(t=moment, rows=None, weight=weight, run=index))
            continue
        serving = raw.get("serving_row")
        serving = serving if serving in (1, 2) else None
        reads.append(
            BoardRead(
                t=moment,
                rows=(parsed[0], parsed[1]),
                serving_row=serving,
                weight=weight,
                run=index,
            )
        )
    return reads


def board_row_names(reads: Sequence[BoardRead]) -> dict[int, str]:
    """The name the reader puts on each row most often over the match."""
    counts: dict[int, dict[str, int]] = {1: {}, 2: {}}
    for read in reads:
        if read.rows is None:
            continue
        for index, row in enumerate(read.rows, start=1):
            if row.name:
                counts[index][row.name] = counts[index].get(row.name, 0) + 1
    return {row: max(counter, key=counter.get) for row, counter in counts.items() if counter}


def orient_rows(read: BoardRead, names: dict[int, str]) -> BoardRead:
    """Swap the rows back when the reader returned them bottom-first.

    The row name is an observation like any other: two names that are each other's row
    mate are a transposed read, and correcting it recovers the read instead of losing it.
    """
    if read.rows is None or len(names) != 2 or names[1] == names[2]:
        return read
    top, bottom = read.rows
    if top.name and bottom.name and top.name == names[2] and bottom.name == names[1]:
        serving = None if read.serving_row is None else 3 - read.serving_row
        return BoardRead(
            t=read.t,
            rows=(bottom, top),
            serving_row=serving,
            weight=read.weight,
            run=read.run,
        )
    return read


def board_observations(
    records: Sequence[dict],
    *,
    fps: float = 1.0,
    weights: Sequence[float] | None = None,
    points_only_boards: bool = False,
) -> list[BoardRead]:
    """Raw reader records -> board reads, rows oriented by the majority row names."""
    reads = board_reads_from_raw(
        records, fps=fps, weights=weights, points_only_boards=points_only_boards
    )
    names = board_row_names(reads)
    return [orient_rows(read, names) for read in reads]


def expand_board_reads(
    reads: Sequence[BoardRead], *, hold_seconds: float = 30.0
) -> list[BoardRead]:
    """One observation per second: seconds between sends inherit the last sent read.

    The reader only sends a crop when the board changed, so the seconds in between carry
    the same board.  They are held for at most ``hold_seconds`` past the send, after which
    the decoder is left with no evidence rather than stale evidence.
    """
    if not reads:
        return []
    expanded: list[BoardRead] = []
    for position, read in enumerate(reads):
        start = read.t
        end = reads[position + 1].t if position + 1 < len(reads) else read.t + 1.0
        moment = start
        while moment < end:
            held = read if moment - start <= hold_seconds else BoardRead(t=moment, rows=None)
            expanded.append(
                BoardRead(
                    t=moment,
                    rows=held.rows,
                    serving_row=held.serving_row,
                    weight=read.weight,
                    run=read.run,
                )
            )
            moment += 1.0
    return expanded


def _row_alignment_score(
    expected: tuple[str, ...], expected_points: str, row: RowRead, points_shown: bool
) -> float:
    """The best score over the layout variants that could have produced ``row``."""
    best = SHAPE_FLOOR
    for seed in (0, 1):
        columns = row.columns[seed:]
        if not columns and not row.points:
            continue
        seed_penalty = SEED_PREFIX_PENALTY if seed else 0.0
        variants = [(columns, row.points, 0.0)] if columns or not row.columns else []
        if row.points:
            # the reader called the current-games box the points box
            variants.append((columns + (row.points,), "", POINTS_AS_GAMES_PENALTY))
        for tokens, points, penalty in variants:
            if len(tokens) == len(expected):
                hidden = 0.0
            elif len(tokens) == len(expected) - 1:
                hidden = HIDDEN_COLUMN_PENALTY  # the board hid the games box (usually 0-0)
            else:
                continue
            score = -seed_penalty - penalty - hidden
            for index, token in enumerate(tokens):
                weight = SET_COLUMN_WEIGHT if index < len(expected) - 1 else GAMES_COLUMN_WEIGHT
                score += weight if token == expected[index] else -weight
            if points:
                score += POINTS_WEIGHT if points == expected_points else -POINTS_WEIGHT
            elif points_shown:
                score -= HIDDEN_COLUMN_PENALTY
            else:
                score += BLANK_POINTS_BONUS
            best = max(best, score)
    return best


def board_emission_score(
    state: ScoreState, read: BoardRead, names: dict[int, str] | None = None
) -> float:
    """Score a hypothesis against one board read, letting the grammar pick the layout."""
    if read.rows is None:
        return 0.0
    # A board only prints the points box while a game is being played from 0-0 upwards; at
    # 0-0 most broadcasts leave it blank, so a blank box is weak evidence for 0-0.
    points_shown = state.points != (0, 0)
    score = 0.0
    for index, row in enumerate(read.rows):
        score += _row_alignment_score(
            state.row_tokens(index), state.point_text(index), row, points_shown
        )
    if read.serving_row in (1, 2):
        actual = _tiebreak_server(state) if state.in_tiebreak else state.server
        score += SERVE_WEIGHT if actual == read.serving_row else -SERVE_WEIGHT
    if names:
        for index, row in enumerate(read.rows, start=1):
            if row.name and names.get(index):
                score += NAME_WEIGHT if row.name == names[index] else -NAME_WEIGHT
    return score * read.weight


def board_seed_states(
    reads: Sequence[BoardRead], rules: MatchRules = DEFAULT_RULES, *, limit: int = 160
) -> list[ScoreState]:
    """States the first reads could describe, under every layout variant."""
    seeds: dict[ScoreState, None] = {}
    for server in (1, 2):
        seeds.setdefault(ScoreState((), (0, 0), (0, 0), server), None)
    for read in reads:
        if read.rows is None:
            continue
        for seed in (0, 1):
            columns = [row.columns[seed:] for row in read.rows]
            points = [row.points for row in read.rows]
            for take_points_as_games in (False, True):
                if take_points_as_games:
                    if not all(points):
                        continue
                    columns = [row.columns[seed:] + (row.points,) for row in read.rows]
                    points = ["", ""]
                if not columns[0] or len(columns[0]) != len(columns[1]):
                    continue
                read_dict = {
                    "g1": columns[0][-1],
                    "g2": columns[1][-1],
                    "p1": points[0] or "0",
                    "p2": points[1] or "0",
                    "sets1": " ".join(columns[0][:-1]),
                    "sets2": " ".join(columns[1][:-1]),
                }
                for server in (1, 2):
                    state = state_from_read(read_dict, server, rules)
                    if state is not None:
                        seeds.setdefault(state, None)
        if len(seeds) >= limit:
            break
    return list(seeds)


# ---------------- decoding -----------------------------------------------------------------


def decode(
    observations: list[tuple[float, dict[str, str] | None, int | None]],
    *,
    beam: int = 96,
    transition_penalty: float = 2.0,
    quick_point_seconds: float = 8.0,
    quick_point_penalty: float = 6.0,
    rules: MatchRules = DEFAULT_RULES,
) -> list[ScoreState | None]:
    """Beam-decode one legal score path over slotted ``score_runs.csv`` observations."""
    if not observations:
        return []
    seeds: dict[ScoreState, float] = {}
    for _, read, _serving in observations[:180]:
        if read is None:
            continue
        for server in (1, 2):
            state = state_from_read(read, server, rules)
            if state is not None:
                seeds.setdefault(state, 0.0)
        if len(seeds) >= beam:
            break
    if not seeds:
        return [None] * len(observations)
    backpointers: list[dict[int, int]] = []
    states_per_step: list[list[tuple[ScoreState, float]]] = []
    current = [(state, 0.0, -1e9) for state in seeds]
    for moment, read, serving in observations:
        scored: dict[ScoreState, tuple[float, int, float]] = {}
        for index, (state, total, last_change) in enumerate(current):
            candidates = [(state, total, last_change)]
            for successor in successors(state, rules):
                penalty = transition_penalty
                if moment - last_change < quick_point_seconds:
                    penalty += quick_point_penalty
                candidates.append((successor, total - penalty, moment))
            for candidate, value, change in candidates:
                value += emission_score(candidate, read, serving)
                existing = scored.get(candidate)
                if existing is None or value > existing[0]:
                    scored[candidate] = (value, index, change)
        ordered = sorted(scored.items(), key=lambda item: -item[1][0])[:beam]
        backpointers.append({position: value[1] for position, (_, value) in enumerate(ordered)})
        states_per_step.append([(state, value[0]) for state, value in ordered])
        current = [(state, value[0], value[2]) for state, value in ordered]
    return _backtrack(states_per_step, backpointers)


def _backtrack(states_per_step, backpointers) -> list[ScoreState | None]:
    path: list[ScoreState | None] = [None] * len(states_per_step)
    position = 0
    for step in range(len(states_per_step) - 1, -1, -1):
        path[step] = states_per_step[step][position][0]
        position = backpointers[step][position]
    return path


def decode_board(
    reads: Sequence[BoardRead],
    *,
    beam: int = 96,
    transition_penalty: float = 12.0,
    quick_point_seconds: float = 8.0,
    quick_point_penalty: float = 6.0,
    rules: MatchRules = DEFAULT_RULES,
    names: dict[int, str] | None = None,
) -> list[ScoreState | None]:
    """Beam-decode one legal score path over raw board reads.

    The transition penalty is far larger than the slotted decoder's because the alignment
    emission is more forgiving: it is what stops a noisy 960-wide read from being spent on
    an invented point.  Measured over the twelve development windows it moves the decoded
    transition count over those windows from 259 to 227 against 206 owner points, and it costs nothing on the
    three finals.
    """
    if not reads:
        return []
    seeds = board_seed_states(reads[:600], rules, limit=max(beam, 160))
    if not seeds:
        return [None] * len(reads)
    emission_cache: dict[tuple[ScoreState, int], float] = {}
    successor_cache: dict[ScoreState, list[ScoreState]] = {}
    backpointers: list[dict[int, int]] = []
    states_per_step: list[list[tuple[ScoreState, float]]] = []
    current = [(state, 0.0, -1e9) for state in seeds]
    for read in reads:
        moment = read.t
        usable = read.rows is not None
        key_run = read.run
        scored: dict[ScoreState, tuple[float, int, float]] = {}
        for index, (state, total, last_change) in enumerate(current):
            candidates = [(state, total, last_change)]
            following = successor_cache.get(state)
            if following is None:
                following = successors(state, rules)
                successor_cache[state] = following
            for successor in following:
                penalty = transition_penalty
                if moment - last_change < quick_point_seconds:
                    penalty += quick_point_penalty
                candidates.append((successor, total - penalty, moment))
            for candidate, value, change in candidates:
                if usable:
                    cache_key = (candidate, key_run)
                    emission = emission_cache.get(cache_key)
                    if emission is None:
                        emission = board_emission_score(candidate, read, names)
                        emission_cache[cache_key] = emission
                    value += emission
                existing = scored.get(candidate)
                if existing is None or value > existing[0]:
                    scored[candidate] = (value, index, change)
        ordered = sorted(scored.items(), key=lambda item: -item[1][0])[:beam]
        backpointers.append({position: value[1] for position, (_, value) in enumerate(ordered)})
        states_per_step.append([(state, value[0]) for state, value in ordered])
        current = [(state, value[0], value[2]) for state, value in ordered]
    return _backtrack(states_per_step, backpointers)


def advance_transitions_into_gaps(
    reads: Sequence[BoardRead],
    path: Sequence[ScoreState | None],
    *,
    settle_seconds: float = 0.0,
    max_pull_back_seconds: float = 1e9,
) -> list[ScoreState | None]:
    """Move each decoded change to the earliest second its evidence allows.

    Viterbi puts a change at the first crop that shows it, which is the *latest* time it
    could have happened.  When the overlay was off screen for half a minute the real change
    is anywhere inside that gap, and a serve inside the gap then carries the previous
    point's score.  Changes are therefore spread evenly across the unobserved gap, starting
    just after the last crop that still showed the old state, which is what the evidence
    actually says: the board changed somewhere in here, m times, in this order.
    """
    out = list(path)
    steps = [
        step
        for step in range(1, len(path))
        if path[step] is not None and path[step - 1] is not None and path[step] != path[step - 1]
    ]
    floor = 0
    index = 0
    while index < len(steps):
        first = steps[index]
        group = 1
        while index + group < len(steps) and steps[index + group] == first + group:
            group += 1
        support = first - 1
        while support >= floor and reads[support].rows is None:
            support -= 1
        if support < floor:
            earliest = floor
        else:
            deadline = max(
                reads[support].t + settle_seconds,
                reads[first].t - max_pull_back_seconds,
            )
            earliest = support + 1
            while earliest < first and reads[earliest].t < deadline:
                earliest += 1
        span = max(0, first - earliest)
        placed = floor
        for offset in range(group):
            position = min(first + offset, max(placed, earliest + offset * span // group))
            for step in range(position, first + offset):
                out[step] = path[first + offset]
            placed = position + 1
        floor = max(floor, placed)
        index += group
    return out


@dataclass
class BoardDecode:
    """A decoded path with a per-second confidence and the age of the evidence behind it."""

    reads: list[BoardRead] = field(default_factory=list)
    path: list[ScoreState | None] = field(default_factory=list)
    confidence: list[float] = field(default_factory=list)
    stale_seconds: list[float] = field(default_factory=list)

    def rows(self) -> list[dict]:
        out = []
        for read, state, confidence, stale in zip(
            self.reads, self.path, self.confidence, self.stale_seconds
        ):
            if state is None:
                continue
            out.append(
                {
                    "t": round(read.t, 3),
                    **state.as_columns(),
                    "score_confidence": round(confidence, 4),
                    "read_age_seconds": round(stale, 1),
                }
            )
        return out


def _stale_seconds(reads: Sequence[BoardRead]) -> list[float]:
    ages = []
    last = None
    for read in reads:
        if read.rows is not None:
            last = read.t
        ages.append(1e6 if last is None else read.t - last)
    return ages


def decode_board_with_confidence(
    reads: Sequence[BoardRead],
    *,
    beam: int = 96,
    ensemble: int = 8,
    dropout: float = 0.25,
    seed: int = 0,
    rules: MatchRules = DEFAULT_RULES,
    names: dict[int, str] | None = None,
    settle_seconds: float = 0.0,
    max_pull_back_seconds: float = 1e9,
    **kwargs,
) -> BoardDecode:
    """Decode, then measure how much of the decode survives dropping a quarter of the reads.

    A state the grammar had to invent because no read supports it changes as soon as the
    reads around it move; a state several independent crops agree on does not.  The share
    of the ensemble that reproduces a second's state is that second's confidence.
    """
    path = advance_transitions_into_gaps(
        reads,
        decode_board(reads, beam=beam, rules=rules, names=names, **kwargs),
        settle_seconds=settle_seconds,
        max_pull_back_seconds=max_pull_back_seconds,
    )
    agreement = [0] * len(reads)
    runs = sorted({read.run for read in reads if read.rows is not None})
    rng = random.Random(seed)
    members = 0
    for _ in range(max(0, ensemble)):
        dropped = {run for run in runs if rng.random() < dropout}
        if len(dropped) == len(runs):
            continue
        sampled = [
            read if read.run not in dropped else BoardRead(t=read.t, rows=None, run=read.run)
            for read in reads
        ]
        member = advance_transitions_into_gaps(
            sampled,
            decode_board(sampled, beam=beam, rules=rules, names=names, **kwargs),
            settle_seconds=settle_seconds,
            max_pull_back_seconds=max_pull_back_seconds,
        )
        members += 1
        for index, (left, right) in enumerate(zip(path, member)):
            if left is not None and right is not None and left == right:
                agreement[index] += 1
    confidence = [
        (count / members) if members else (1.0 if state is not None else 0.0)
        for count, state in zip(agreement, path)
    ]
    return BoardDecode(
        reads=list(reads),
        path=path,
        confidence=confidence,
        stale_seconds=_stale_seconds(reads),
    )


# ---------------- output -------------------------------------------------------------------


def score_runs_observations(
    rows: list[dict], serving_rows: dict[float, int] | None = None
) -> list[tuple[float, dict[str, str] | None, int | None]]:
    """Expand collapsed score runs back to one observation per second."""
    observations = []
    for row in rows:
        start, end = float(row["t_start"]), float(row["t_end"])
        read = parse_read(row)
        moment = start
        while moment < end:
            serving = (serving_rows or {}).get(round(moment))
            observations.append((moment, read, serving))
            moment += 1.0
    return observations


def point_boundaries(
    observations: list[tuple[float, dict[str, str] | None, int | None]],
    path: list[ScoreState | None],
) -> list[dict]:
    """Times at which the decoded score advanced by one point, with the state before it."""
    boundaries = []
    for index in range(1, len(path)):
        if path[index] is None or path[index - 1] is None:
            continue
        if path[index] != path[index - 1]:
            boundaries.append(
                {
                    "t": observations[index][0],
                    **{
                        f"before_{key}": value
                        for key, value in path[index - 1].as_columns().items()
                    },
                }
            )
    return boundaries


DECODE_COLUMNS = [
    "t",
    *ScoreState((), (0, 0), (0, 0), 1).as_columns(),
    "score_confidence",
    "read_age_seconds",
]


def write_decode(
    out_dir: Path, observations, path, *, source: str, rules: MatchRules = DEFAULT_RULES
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "score_state_v1.csv"
    columns = ["t", *ScoreState((), (0, 0), (0, 0), 1).as_columns()]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for (moment, _, _), state in zip(observations, path):
            if state is None:
                continue
            writer.writerow({"t": round(moment, 3), **state.as_columns()})
    boundaries = point_boundaries(observations, path)
    (out_dir / "score_state_v1.json").write_text(
        json.dumps(
            {
                "schema": "score_grammar_decode_v1",
                "match_rules": asdict(rules),
                "source": source,
                "observations": len(observations),
                "decoded": sum(state is not None for state in path),
                "point_transitions": len(boundaries),
                "boundaries": boundaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return csv_path


def write_board_decode(
    out_dir: Path,
    decode_result: BoardDecode,
    *,
    source: str,
    rules: MatchRules = DEFAULT_RULES,
    points_only_boards: bool = False,
) -> Path:
    """Write the decoded path with its confidence, plus a summary of what it abstains on."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "score_state_v1.csv"
    rows = decode_result.rows()
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DECODE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    transitions = 0
    previous = None
    for state in decode_result.path:
        if state is not None and previous is not None and state != previous:
            transitions += 1
        if state is not None:
            previous = state
    completed = ""
    for state in reversed(decode_result.path):
        if state is not None:
            completed = ";".join(f"{a}-{b}" for a, b in state.completed)
            break
    (out_dir / "score_state_v1.json").write_text(
        json.dumps(
            {
                "schema": "score_grammar_decode_v2",
                "match_rules": asdict(rules),
                "points_only_boards": points_only_boards,
                "source": source,
                "observations": len(decode_result.path),
                "usable_reads": sum(read.rows is not None for read in decode_result.reads),
                "decoded": sum(state is not None for state in decode_result.path),
                "point_transitions": transitions,
                "completed_sets": completed,
                "mean_confidence": (
                    round(sum(decode_result.confidence) / len(decode_result.confidence), 4)
                    if decode_result.confidence
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return csv_path


def read_raw_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-runs", type=Path, help="slotted reader output (v1 path)")
    parser.add_argument("--vlm-reads", type=Path, help="raw reader output (v2 path)")
    parser.add_argument(
        "--slotted",
        action="store_true",
        help="require the explicitly supplied slotted score_runs.csv input",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--beam", type=int, default=96)
    parser.add_argument("--best-of", type=int, choices=(3, 5), default=5)
    parser.add_argument("--tiebreak-target", type=int, choices=(7, 10), default=7)
    parser.add_argument("--deciding-tiebreak-target", type=int, choices=(7, 10), default=10)
    parser.add_argument("--ensemble", type=int, default=8)
    parser.add_argument(
        "--points-only-boards",
        action="store_true",
        help="candidate support for visible point scores with hidden game columns; requires composed segmentation validation",
    )
    parser.add_argument("--fps", type=float, default=1.0)
    args = parser.parse_args()
    if not args.score_runs and not args.vlm_reads:
        parser.error("one of --score-runs or --vlm-reads is required")
    if args.score_runs and args.vlm_reads:
        parser.error("choose exactly one of --score-runs or --vlm-reads")
    if args.slotted and not args.score_runs:
        parser.error("--slotted requires --score-runs")
    if args.points_only_boards and not args.vlm_reads:
        parser.error("--points-only-boards requires --vlm-reads")
    rules = MatchRules(args.best_of, args.tiebreak_target, args.deciding_tiebreak_target)
    if args.vlm_reads:
        records = read_raw_records(args.vlm_reads)
        reads = expand_board_reads(
            board_observations(records, fps=args.fps, points_only_boards=args.points_only_boards)
        )
        names = board_row_names(reads)
        result = decode_board_with_confidence(
            reads, beam=args.beam, ensemble=args.ensemble, rules=rules, names=names
        )
        csv_path = write_board_decode(
            args.out,
            result,
            source=str(args.vlm_reads),
            rules=rules,
            points_only_boards=args.points_only_boards,
        )
        decoded = sum(state is not None for state in result.path)
        print(f"{len(reads)} observations, {decoded} decoded -> {csv_path}")
        return 0
    with args.score_runs.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    observations = score_runs_observations(rows)
    path = decode(observations, beam=args.beam, rules=rules)
    csv_path = write_decode(args.out, observations, path, source=str(args.score_runs), rules=rules)
    changes = sum(
        1
        for index in range(1, len(path))
        if path[index] is not None and path[index] != path[index - 1]
    )
    print(f"{len(observations)} observations, {changes} point transitions -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
