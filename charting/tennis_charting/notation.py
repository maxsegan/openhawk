"""Parser for Match Charting Project shot-by-shot notation.

Pure and CSV-agnostic: it turns the ``1st``/``2nd`` code strings of a single point into a
structured :class:`Point` (serves, rally shots, outcome, and which *role* — server or
returner — won). Resolving roles to absolute player numbers needs the ``Svr`` column and
happens in :mod:`tennis_charting.mcp_parse`.

Grammar reference: ``charting/NOTATION.md`` (distilled from the MCP spreadsheet's
Instructions tab, which is authoritative).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# --- character classes ------------------------------------------------------------------

SERVE_DIRECTIONS = {"4": "wide", "5": "body", "6": "t", "0": "unknown"}

# Shot-type letter -> (canonical name, wing). Wing groups fh-/bh-family for split stats.
SHOT_TYPES: dict[str, tuple[str, str]] = {
    "f": ("forehand", "fh"),
    "b": ("backhand", "bh"),
    "r": ("fh_slice", "fh"),
    "s": ("bh_slice", "bh"),
    "v": ("fh_volley", "fh"),
    "z": ("bh_volley", "bh"),
    "o": ("overhead", "fh"),
    "p": ("bh_overhead", "bh"),
    "u": ("fh_drop", "fh"),
    "y": ("bh_drop", "bh"),
    "l": ("fh_lob", "fh"),
    "m": ("bh_lob", "bh"),
    "h": ("fh_halfvolley", "fh"),
    "i": ("bh_halfvolley", "bh"),
    "j": ("fh_swingvolley", "fh"),
    "k": ("bh_swingvolley", "bh"),
    "t": ("trick", "other"),
    "q": ("unknown", "other"),
}

# Terminal error/fault types. NOTE: "!" (shank) is deliberately NOT here — in the data it is
# overwhelmingly a mid-rally shot annotation ("f!28" = a shanked forehand that stayed in
# play), so it is handled as a shot modifier below and only ends the point via a following
# @/# or an explicit error letter (e.g. "f!1w@").
ERROR_TYPES = {"n": "net", "w": "wide", "d": "deep", "x": "wide_deep", "e": "unknown"}
FAULT_TYPES = {**ERROR_TYPES, "g": "foot_fault"}

MODIFIERS = {"+": "approach", "-": "at_net", "=": "at_baseline", ";": "net_cord",
             "^": "stop_volley", "!": "shank"}

_SHOT_LETTERS = set(SHOT_TYPES)
_DIR_DIGITS = set("0123")
_DEPTH_DIGITS = set("0789")


class Outcome(str, Enum):
    ACE = "ace"
    UNRETURNABLE = "unreturnable"
    DOUBLE_FAULT = "double_fault"
    WINNER = "winner"
    FORCED_ERROR = "forced_error"
    UNFORCED_ERROR = "unforced_error"
    SERVER_AWARDED = "server_awarded"       # 'S'
    RETURNER_AWARDED = "returner_awarded"   # 'R'
    PENALTY_SERVER = "penalty_server"       # 'P' -> returner wins
    PENALTY_RETURNER = "penalty_returner"   # 'Q' -> server wins
    TIME_VIOLATION = "time_violation"       # 'V' as a lost first serve is not point-ending
    CHALLENGE = "challenge_incorrect"       # 'C'
    UNKNOWN = "unknown"


class Role(str, Enum):
    SERVER = "server"
    RETURNER = "returner"


@dataclass
class Serve:
    number: int                 # 1 or 2
    direction: str | None       # wide/body/t/unknown/None
    serve_and_volley: bool = False
    lets: int = 0
    in_play: bool = False       # landed in (rally followed, or ace/unret)
    fault_type: str | None = None  # net/wide/deep/wide_deep/foot_fault/shank/unknown/None


@dataclass
class Shot:
    index: int                  # 0-based position in the rally (0 = return)
    role: Role                  # who hit it
    letter: str
    shot_type: str              # canonical name
    wing: str                   # fh / bh / other
    direction: int | None = None   # 0..3 / None
    depth: int | None = None       # 0/7/8/9 / None (returns)
    approach: bool = False
    at_net: bool = False
    at_baseline: bool = False
    net_cord: bool = False
    stop_volley: bool = False
    shank: bool = False
    error_type: str | None = None  # set on the errored terminal shot
    error_forced: bool | None = None  # True forced / False unforced / None if not an error
    winner: bool = False


@dataclass
class Point:
    serves: list[Serve] = field(default_factory=list)
    shots: list[Shot] = field(default_factory=list)
    outcome: Outcome = Outcome.UNKNOWN
    winner_role: Role | None = None   # server / returner / None if unresolved
    rally_len: int = 0                # MCP convention: shots incl. serve, excl. the error shot
    warnings: list[str] = field(default_factory=list)
    raw_first: str = ""
    raw_second: str = ""

    @property
    def resolved(self) -> bool:
        """True if a definite winner_role was determined (excludes challenges/admin gaps)."""
        return self.winner_role is not None


# --- scanning helpers -------------------------------------------------------------------


def _parse_shot(code: str, i: int, index: int, role: Role) -> tuple[Shot, int, bool]:
    """Parse one rally shot starting at ``code[i]``.

    Returns (shot, next_index, terminal) where ``terminal`` means this shot ended the point
    (winner, error, or challenge). Assumes ``code[i]`` is a shot letter.
    """
    letter = code[i]
    name, wing = SHOT_TYPES[letter]
    shot = Shot(index=index, role=role, letter=letter, shot_type=name, wing=wing)
    i += 1
    n = len(code)

    def consume_modifiers(j: int) -> int:
        while j < n and code[j] in MODIFIERS:
            m = MODIFIERS[code[j]]
            setattr(shot, m, True)
            j += 1
        return j

    i = consume_modifiers(i)
    # up to two digits: first -> direction, second -> depth (position + value based)
    digits: list[str] = []
    while i < n and code[i].isdigit() and len(digits) < 2:
        digits.append(code[i])
        i += 1
    for d in digits:
        val = int(d)
        if d in _DIR_DIGITS and shot.direction is None:
            shot.direction = val
        elif d in _DEPTH_DIGITS:
            shot.depth = val
        elif shot.direction is None:
            shot.direction = val
        else:
            shot.depth = val
    i = consume_modifiers(i)

    # terminal markers
    if i < n and code[i] == "*":
        shot.winner = True
        return shot, i + 1, True
    # 'C' (incorrect challenge) is left for _parse_rally to turn into a CHALLENGE outcome.
    if i < n and code[i] in ERROR_TYPES:
        shot.error_type = ERROR_TYPES[code[i]]
        i += 1
        if i < n and code[i] in "@#":
            shot.error_forced = code[i] == "#"
            i += 1
        else:
            shot.error_forced = None  # error letter without @/# (rare); treat as error, force unknown
        return shot, i, True
    if i < n and code[i] in "@#":
        shot.error_forced = code[i] == "#"  # bare forced/unforced, no error letter
        return shot, i + 1, True
    return shot, i, False


def _parse_serve_and_rally(code: str, number: int, point: Point, first_hitter: Role) -> None:
    """Parse a cell that holds a serve (number 1 or 2) and, if in, the rally + ending."""
    code = code.strip()
    n = len(code)
    i = 0
    serve = Serve(number=number, direction=None)

    # leading lets
    while i < n and code[i] == "c":
        serve.lets += 1
        i += 1

    if i < n and code[i] == "V":  # time violation loses this serve
        serve.fault_type = "time_violation"
        point.serves.append(serve)
        point.outcome = Outcome.TIME_VIOLATION
        return

    if i < n and code[i] in SERVE_DIRECTIONS:
        serve.direction = SERVE_DIRECTIONS[code[i]]
        i += 1
    if i < n and code[i] == "+":
        serve.serve_and_volley = True
        i += 1

    # serve resolution
    if i >= n:
        serve.in_play = True  # bare direction, rally not recorded
        point.serves.append(serve)
        return
    ch = code[i]
    if ch == "*":
        serve.in_play = True
        point.serves.append(serve)
        point.outcome = Outcome.ACE
        point.winner_role = Role.SERVER
        return
    if ch == "#":
        serve.in_play = True
        point.serves.append(serve)
        point.outcome = Outcome.UNRETURNABLE
        point.winner_role = Role.SERVER
        return
    if ch in FAULT_TYPES and ch not in _SHOT_LETTERS:
        serve.fault_type = FAULT_TYPES[ch]
        point.serves.append(serve)
        return  # serve out; caller decides double-fault vs second serve
    if ch == "!":
        serve.fault_type = "shank"
        point.serves.append(serve)
        return

    # otherwise the serve is in and a rally follows
    serve.in_play = True
    point.serves.append(serve)
    _parse_rally(code, i, point, first_hitter)


def _parse_rally(code: str, i: int, point: Point, first_hitter: Role) -> None:
    n = len(code)
    idx = 0
    role = first_hitter
    while i < n:
        ch = code[i]
        if ch in _SHOT_LETTERS:
            shot, i, terminal = _parse_shot(code, i, idx, role)
            point.shots.append(shot)
            idx += 1
            role = Role.SERVER if role == Role.RETURNER else Role.RETURNER
            if terminal:
                _finalize_from_last_shot(point)
                return
        elif ch in "@#":
            # bare terminal error attached to the previous shot (defensive)
            if point.shots:
                point.shots[-1].error_forced = ch == "#"
                _finalize_from_last_shot(point)
            return
        elif ch == "C":
            point.outcome = Outcome.CHALLENGE
            return
        elif ch in " '.":  # known stray characters in a handful of charts
            i += 1
        else:
            point.warnings.append(f"unparsed char {ch!r} at {i} in {code!r}")
            i += 1
    # rally ran out with no explicit terminal (truncated chart)
    if point.shots:
        point.warnings.append(f"rally without terminal marker: {code!r}")


def _finalize_from_last_shot(point: Point) -> None:
    last = point.shots[-1]
    hitter = last.role
    other = Role.SERVER if hitter == Role.RETURNER else Role.RETURNER
    if last.winner:
        point.outcome = Outcome.WINNER
        point.winner_role = hitter
    elif last.error_type is not None or last.error_forced is not None:
        forced = last.error_forced
        point.outcome = Outcome.FORCED_ERROR if forced else Outcome.UNFORCED_ERROR
        point.winner_role = other
    # challenge handled by caller


def _compute_rally_len(point: Point) -> int:
    """MCP rallyCount convention: shots including the serve, excluding the final error shot."""
    if point.outcome in (Outcome.DOUBLE_FAULT, Outcome.SERVER_AWARDED, Outcome.RETURNER_AWARDED,
                          Outcome.PENALTY_SERVER, Outcome.PENALTY_RETURNER, Outcome.TIME_VIOLATION,
                          Outcome.UNKNOWN):
        return 0
    serve_in = any(s.in_play for s in point.serves)
    n = 1 if serve_in else 0
    for shot in point.shots:
        is_error = shot.error_type is not None or shot.error_forced is not None
        if not is_error:
            n += 1
    return n


# --- public API -------------------------------------------------------------------------

_ADMIN = {"S": Outcome.SERVER_AWARDED, "R": Outcome.RETURNER_AWARDED,
          "P": Outcome.PENALTY_SERVER, "Q": Outcome.PENALTY_RETURNER}
_ADMIN_WINNER = {"S": Role.SERVER, "R": Role.RETURNER, "P": Role.RETURNER, "Q": Role.SERVER}


def parse_point(first: str, second: str = "") -> Point:
    """Parse the ``1st``/``2nd`` code strings of one point into a :class:`Point`."""
    first = (first or "").strip()
    second = (second or "").strip()
    point = Point(raw_first=first, raw_second=second)

    if first in _ADMIN and not second:
        point.outcome = _ADMIN[first]
        point.winner_role = _ADMIN_WINNER[first]
        return point
    if not first:
        point.warnings.append("empty first-serve cell")
        return point

    _parse_serve_and_rally(first, 1, point, first_hitter=Role.RETURNER)

    first_faulted = point.serves and point.serves[0].fault_type is not None
    if first_faulted:
        if second:
            _parse_serve_and_rally(second, 2, point, first_hitter=Role.RETURNER)
            second_faulted = len(point.serves) > 1 and point.serves[1].fault_type is not None
            if second_faulted and point.outcome == Outcome.UNKNOWN:
                point.outcome = Outcome.DOUBLE_FAULT
                point.winner_role = Role.RETURNER
        elif point.outcome == Outcome.UNKNOWN:
            # first fault, no second recorded: treat as double fault (server lost the point)
            point.outcome = Outcome.DOUBLE_FAULT
            point.winner_role = Role.RETURNER
    elif second:
        point.warnings.append("second-serve cell present but first serve did not fault")

    point.rally_len = _compute_rally_len(point)
    return point
