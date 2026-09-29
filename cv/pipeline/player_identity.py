"""Name the two players of a match once, and keep the naming stable across its points.

Owner decision 5 (2026-09-01): "Player identity by name is required, not just near/far
side. Identify each player once per match and keep the assignment stable through ends
changes." Nothing on the default path did this; the only naming code in the repo
(``camera_metric_refine``) decided names from body height and declared itself safe when
the two rostered heights were within 3 cm, which is not a decision at all. Height is not
used here.

The method has three independent parts, and the artifact records which one carried each
claim so a consumer can tell a measured fact from a convention:

1. APPEARANCE. Each point contributes two appearance vectors, one per side, averaged over
   that side's tracked frames. Across the match the points are clustered into exactly two
   identities under the constraint that the two sides of one point can never be the same
   identity, so a point is only ever assigned an orientation: near=A/far=B, or the swap.
   The margin between the two orientations is reported per point.
2. ENDS PARITY. Players change ends at the end of the first, third and every subsequent
   alternate game of each set (ITF rule 10). Given a per-point game index, this fixes the
   near/far identity of every point from the first one alone, with no image evidence.
   Parity is a cross-check on appearance, not a replacement: it needs a game index, which
   the frame corpora do not carry.
3. ROSTER. Match ids carry the two player names. Attaching a name to identity A rather
   than identity B needs an anchor that neither appearance nor parity can supply, so the
   roster order is used and stated as a convention (``name_source`` is
   ``roster_order_unanchored``) unless the caller pins it with an explicit anchor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA = "tennis.player_identity.v1"
IDENTITIES = ("A", "B")
GENDER_TOKENS = ("m", "w")
# Surname particles, so "alcaraz_de_minaur" splits as alcaraz / de minaur.
PARTICLES = frozenset(
    {"de", "del", "della", "di", "da", "dos", "van", "von", "der", "den", "la", "le", "al"}
)


# --------------------------------------------------------------------------- roster


@dataclass(frozen=True)
class Roster:
    names: tuple[str, str] | None
    source: str
    confidence: str  # "high" when the split is unambiguous, "low" when it is a guess


def roster_from_match_id(match_id: str) -> Roster:
    """Two player names from a source name such as ``wim2025f_m_sinner_alcaraz``.

    Two id shapes exist in the corpora: an event id followed by a gender token and two
    surnames, and a tour id followed by a numeric key and two full names. The split point
    is the last gender or numeric token; the remaining tokens are divided in half, with
    surname particles moved onto the following name.
    """
    tokens = match_id.split("_")
    marker = -1
    shape = ""
    for index, token in enumerate(tokens):
        if token in GENDER_TOKENS:
            marker, shape = index, "surnames"
        elif token.isdigit():
            marker, shape = index, "full_names"
    names = tokens[marker + 1 :]
    if marker < 0 or len(names) < 2:
        return Roster(None, "match_id", "low")
    per_name = 1 if shape == "surnames" else 2
    if len(names) == 2 * per_name:
        split = per_name
        confidence = "high"
    else:
        # An extra token belongs to whichever name owns the particle; otherwise assume the
        # first player has the ordinary-length name.
        split = per_name
        confidence = "low"
        for index in range(1, len(names) - 1):
            if names[index] in PARTICLES:
                split = index
                break
    first = " ".join(names[:split])
    second = " ".join(names[split:])
    if not first or not second:
        return Roster(None, "match_id", "low")
    return Roster((first, second), "match_id", confidence)


# --------------------------------------------------------------------------- ends parity


TIEBREAK_CHANGEOVER_POINTS = 6
TIEBREAK_GAMES = 13
ASSUMED_TIEBREAK_POINTS = 12


def changeovers_in_tiebreak(points_played: int) -> int:
    """Ends changes inside a tiebreak game: one after every six points (ITF rule 10 b).

    This is the part of rule 10 the first implementation left out, and it is not a detail:
    a tiebreak won 7-4 is eleven points and therefore ONE change of ends, which flips the
    parity of every point after it. Measured on the Roland Garros 2025 final, adding it
    moves appearance/parity agreement from 0.62 to 0.98 over 569 attempts.
    """
    return max(0, int(points_played)) // TIEBREAK_CHANGEOVER_POINTS


def changeovers_before(
    completed_sets: list[int],
    games_in_current_set: int,
    *,
    tiebreak_points: int = 0,
    completed_tiebreak_points: list[int] | None = None,
) -> int:
    """Number of end changes that have happened before a point (ITF rule 10).

    Ends change after the first, third and every subsequent alternate game of each set,
    which is ``ceil(games / 2)`` changes for a set of ``games`` completed games. A set
    that ends on an even total leaves the next change to after the first game of the
    following set, which the per-set count reproduces.

    A tiebreak is one game by that count but carries its own changes every six points.
    ``tiebreak_points`` is the number of points played so far in the tiebreak now in
    progress; ``completed_tiebreak_points`` is the final point total of each earlier set
    that reached one, positionally aligned with ``completed_sets``.
    """
    total = sum((games + 1) // 2 for games in completed_sets)
    finished = list(completed_tiebreak_points or [])
    for index, games in enumerate(completed_sets):
        if games != TIEBREAK_GAMES:
            continue
        played = finished[index] if index < len(finished) else ASSUMED_TIEBREAK_POINTS
        total += changeovers_in_tiebreak(played)
    return total + (games_in_current_set + 1) // 2 + changeovers_in_tiebreak(tiebreak_points)


def ends_parity(points: list[dict]) -> dict[str, int]:
    """clip -> 0 when the point is played from the anchor ends, 1 when swapped.

    Each point needs ``set_index`` (0-based) and ``games_completed_in_set``, plus
    ``set_lengths`` (completed games per earlier set), and optionally ``tiebreak_points``
    and ``completed_tiebreak_points``. The first point is the anchor.
    """
    parities: dict[str, int] = {}
    anchor: int | None = None
    for point in points:
        completed = list(point.get("set_lengths") or [])
        count = changeovers_before(
            completed,
            int(point["games_completed_in_set"]),
            tiebreak_points=int(point.get("tiebreak_points") or 0),
            completed_tiebreak_points=point.get("completed_tiebreak_points"),
        )
        if anchor is None:
            anchor = count
        parities[point["clip"]] = (count - anchor) % 2
    return parities


# --------------------------------------------------------------------------- descriptor

TORSO_FRACTION = 0.60
LAB_BINS = (4, 6, 6)
COLOUR_WEIGHT = 0.5


def torso_box(
    box: tuple[float, float, float, float], *, fraction: float = TORSO_FRACTION
) -> tuple[float, float, float, float]:
    """The upper ``fraction`` of a person box, in the same (native) pixels it came in.

    A whole-person crop is mostly legs, court and shadow; the shirt is the part that tells
    two players apart, and it lives in the upper 60% of the box together with the head and
    the racket arm.
    """
    x0, y0, x1, y1 = (float(value) for value in box)
    return (x0, y0, x1, y0 + fraction * (y1 - y0))


def shirt_box(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """The shirt inside a person box: the middle 60% across, 18% to 50% down.

    ``torso_box`` still holds a lot of court either side of a player and above their head,
    and on clay that background is a strong orange that swamps a colour histogram. This
    tighter window is almost all fabric whatever the player is doing.
    """
    x0, y0, x1, y1 = (float(value) for value in box)
    width, height = x1 - x0, y1 - y0
    return (x0 + 0.20 * width, y0 + 0.18 * height, x1 - 0.20 * width, y0 + 0.50 * height)


def chroma_histogram(crop_bgr, *, bins: int = 12) -> np.ndarray:
    """Hellinger-scaled CIELAB a/b histogram, with lightness dropped.

    The near player is lit and exposed differently from the far player, and on a five-hour
    match the light changes under the same shirt. Lightness carries most of that change and
    hue carries most of the kit, so dropping L makes the histogram travel better between the
    two ends of the court than the full three-axis one; it does not make it exposure-proof.
    """
    import cv2

    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros(bins * bins, dtype=np.float32)
    lab = cv2.cvtColor(np.ascontiguousarray(crop_bgr), cv2.COLOR_BGR2LAB)
    histogram = cv2.calcHist([lab], [1, 2], None, [bins, bins], [0, 256, 0, 256]).ravel()
    total = float(histogram.sum())
    if total <= 0:
        return np.zeros(histogram.size, dtype=np.float32)
    return np.sqrt(histogram / total).astype(np.float32)


def torso_colour_histogram(crop_bgr, *, bins: tuple[int, int, int] = LAB_BINS) -> np.ndarray:
    """Hellinger-scaled CIELAB histogram of a BGR crop.

    CIELAB rather than RGB or HSV because equal distances in it are meant to be equal
    perceived colour differences, so two shirt whites under different court lighting stay
    closer to each other than either is to a blue shirt. The square root turns the L1
    histogram into a vector whose dot product is the Bhattacharyya coefficient, which is
    what the cosine distance used everywhere else in this module then measures.
    """
    import cv2

    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros(int(np.prod(bins)), dtype=np.float32)
    lab = cv2.cvtColor(np.ascontiguousarray(crop_bgr), cv2.COLOR_BGR2LAB)
    histogram = cv2.calcHist([lab], [0, 1, 2], None, list(bins), [0, 256, 0, 256, 0, 256]).ravel()
    total = float(histogram.sum())
    if total <= 0:
        return np.zeros(histogram.size, dtype=np.float32)
    return np.sqrt(histogram / total).astype(np.float32)


def combine_descriptor(
    parts: dict[str, np.ndarray], *, weights: dict[str, float] | None = None
) -> np.ndarray:
    """L2-normalise each named part, weight it, concatenate, and L2-normalise the whole.

    Normalising the parts first stops the longer one (a 384-d embedding against a 144-bin
    histogram) from setting the scale by its dimension alone.
    """
    weights = weights or {}
    pieces = []
    for name, vector in parts.items():
        pieces.append(
            _normalise(np.asarray(vector, dtype=np.float32)) * float(weights.get(name, 1.0))
        )
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    return _normalise(np.concatenate(pieces))


# --------------------------------------------------------------------------- appearance


@dataclass(frozen=True)
class PointAppearance:
    clip: str
    near: np.ndarray | None = None
    far: np.ndarray | None = None
    near_frames: int = 0
    far_frames: int = 0

    def vectors(self) -> dict[str, np.ndarray]:
        return {
            side: vector
            for side, vector in (("near", self.near), ("far", self.far))
            if vector is not None
        }


@dataclass
class ClusterResult:
    orientation: dict[str, str] = field(default_factory=dict)  # clip -> "AB" or "BA"
    margin: dict[str, float] = field(default_factory=dict)
    centroids: dict[str, np.ndarray] = field(default_factory=dict)
    within: float = float("nan")
    between: float = float("nan")
    points_used: int = 0
    groups: dict[str, str] = field(default_factory=dict)  # clip -> clustering group
    group_link_margin: dict[str, float] = field(default_factory=dict)


def _normalise(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-6 else vector


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.clip(np.dot(_normalise(a), _normalise(b)), -1.0, 1.0))


def _cluster_group(
    usable: list[PointAppearance], *, iterations: int
) -> tuple[dict[str, str], dict[str, float], dict[str, np.ndarray]]:
    """Two-means over one group of points, near and far forced apart inside each point."""
    # Seed on the point whose two sides look least alike: the cleanest separated pair.
    seed = max(usable, key=lambda p: _distance(p.near, p.far))
    centroids = {"A": _normalise(seed.near), "B": _normalise(seed.far)}
    orientation: dict[str, str] = {}
    margins: dict[str, float] = {}
    for _ in range(iterations):
        assignment: dict[str, str] = {}
        margins = {}
        for point in usable:
            straight = _distance(point.near, centroids["A"]) + _distance(point.far, centroids["B"])
            swapped = _distance(point.near, centroids["B"]) + _distance(point.far, centroids["A"])
            assignment[point.clip] = "AB" if straight <= swapped else "BA"
            margins[point.clip] = round(abs(straight - swapped) / 2.0, 4)
        if assignment == orientation:
            break
        orientation = assignment
        grouped: dict[str, list[np.ndarray]] = {"A": [], "B": []}
        for point in usable:
            near_identity, far_identity = (
                ("A", "B") if orientation[point.clip] == "AB" else ("B", "A")
            )
            grouped[near_identity].append(_normalise(point.near))
            grouped[far_identity].append(_normalise(point.far))
        centroids = {
            key: _normalise(np.mean(np.stack(vectors), axis=0))
            for key, vectors in grouped.items()
            if vectors
        }
        if len(centroids) < 2:
            break
    return orientation, margins, centroids


def cluster_two_identities(
    points: list[PointAppearance],
    *,
    iterations: int = 50,
    groups: dict[str, str] | None = None,
) -> ClusterResult:
    """Two identities over a match, with near and far forced apart inside each point.

    ``groups`` maps a clip to a clustering group, normally the set number. Players change
    kit between sets often enough that one pair of centroids across a whole match is the
    wrong model; clustering runs inside each group and the groups are then chained by
    whichever labelling of the later group's centroids sits closer to the earlier one.
    That chain is the part a kit change breaks, so its margin is reported per group and
    ends parity is what actually carries identity across a change (see ``resolve_match``).
    """
    usable = [p for p in points if p.near is not None and p.far is not None]
    result = ClusterResult(points_used=len(usable))
    if not usable:
        return result
    keys = groups or {}
    result.groups = {point.clip: str(keys.get(point.clip, "")) for point in points}
    partitions: dict[str, list[PointAppearance]] = {}
    for point in usable:
        partitions.setdefault(result.groups[point.clip], []).append(point)

    orientation: dict[str, str] = {}
    margins: dict[str, float] = {}
    reference: dict[str, np.ndarray] | None = None
    for group in sorted(partitions):
        members = partitions[group]
        local, local_margins, centroids = _cluster_group(members, iterations=iterations)
        if len(centroids) < 2:
            continue
        flip = False
        if reference is not None:
            straight = _distance(centroids["A"], reference["A"]) + _distance(
                centroids["B"], reference["B"]
            )
            swapped = _distance(centroids["A"], reference["B"]) + _distance(
                centroids["B"], reference["A"]
            )
            flip = swapped < straight
            result.group_link_margin[group] = round(abs(straight - swapped) / 2.0, 4)
        for clip, value in local.items():
            orientation[clip] = value[::-1] if flip else value
        margins.update(local_margins)
        if flip:
            centroids = {"A": centroids["B"], "B": centroids["A"]}
        reference = centroids

    result.orientation = orientation
    result.margin = margins
    if orientation:
        grouped: dict[str, list[np.ndarray]] = {"A": [], "B": []}
        for point in usable:
            if point.clip not in orientation:
                continue
            near_identity, far_identity = (
                ("A", "B") if orientation[point.clip] == "AB" else ("B", "A")
            )
            grouped[near_identity].append(_normalise(point.near))
            grouped[far_identity].append(_normalise(point.far))
        result.centroids = {
            key: _normalise(np.mean(np.stack(vectors), axis=0))
            for key, vectors in grouped.items()
            if vectors
        }
    if len(result.centroids) == 2:
        within_distances, between_distances = [], []
        for point in usable:
            if point.clip not in orientation:
                continue
            near_identity, far_identity = (
                ("A", "B") if orientation[point.clip] == "AB" else ("B", "A")
            )
            within_distances.append(_distance(point.near, result.centroids[near_identity]))
            within_distances.append(_distance(point.far, result.centroids[far_identity]))
            between_distances.append(_distance(point.near, result.centroids[far_identity]))
            between_distances.append(_distance(point.far, result.centroids[near_identity]))
        result.within = round(float(np.mean(within_distances)), 4)
        result.between = round(float(np.mean(between_distances)), 4)
    # Sides seen alone still get an identity, from the nearer centroid.
    for point in points:
        if point.clip in result.orientation or len(result.centroids) < 2:
            continue
        vectors = point.vectors()
        if not vectors:
            continue
        side, vector = next(iter(vectors.items()))
        nearer = min(IDENTITIES, key=lambda key: _distance(vector, result.centroids[key]))
        other = "B" if nearer == "A" else "A"
        result.orientation[point.clip] = (
            f"{nearer}{other}" if side == "near" else f"{other}{nearer}"
        )
        result.margin[point.clip] = 0.0
    return result


def remove_side_bias(points: list[PointAppearance]) -> list[PointAppearance]:
    """Subtract each side's own mean descriptor, then renormalise.

    A near player is twice as tall in pixels as a far player, sharper, and lit from a
    different angle; measured on the three finals that shared near/far difference is larger
    than the difference between the two players, so a descriptor clustered raw splits into
    "near" and "far" rather than into two people and scores chance against ends parity.
    The near mean and the far mean over a match both contain one of each player in equal
    measure, so removing them cancels the side and keeps the person.
    """
    means = {}
    for side in ("near", "far"):
        stack = [getattr(point, side) for point in points if getattr(point, side) is not None]
        if stack:
            means[side] = np.mean(np.stack([np.asarray(v, np.float32) for v in stack]), axis=0)
    out = []
    for point in points:
        values = {}
        for side in ("near", "far"):
            vector = getattr(point, side)
            values[side] = (
                None
                if vector is None
                else _normalise(np.asarray(vector, np.float32) - means.get(side, 0.0))
            )
        out.append(
            PointAppearance(
                clip=point.clip,
                near=values["near"],
                far=values["far"],
                near_frames=point.near_frames,
                far_frames=point.far_frames,
            )
        )
    return out


# --------------------------------------------------------------------------- witnesses


def _tiebreak_point(value) -> int:
    text = str(value or "").strip()
    return int(text) if text.isdigit() else 0


def parity_from_scores(points: list[dict]) -> dict[str, int]:
    """clip -> ends parity, from the decoded (set, games, points) of ``score_grammar``.

    Each point needs ``clip``, ``set_number`` (1-based), ``games_1``, ``games_2``,
    ``points_1``, ``points_2`` and ``completed_sets`` as the decoder writes them
    ("6-4;7-6"). A set at six games all is in its tiebreak, so its point columns are read
    as tiebreak points; the longest tiebreak seen in a set is taken as that tiebreak's
    length once the set is over, because a tiebreak only ever grows. The first point
    anchors the parity, exactly as ``ends_parity`` does for a hand-built game index.
    """
    longest: dict[int, int] = {}
    for point in points:
        games = (int(point.get("games_1") or 0), int(point.get("games_2") or 0))
        if games != (6, 6):
            continue
        played = _tiebreak_point(point.get("points_1")) + _tiebreak_point(point.get("points_2"))
        set_number = int(point.get("set_number") or 0)
        longest[set_number] = max(longest.get(set_number, 0), played)

    prepared = []
    for point in points:
        completed = [
            sum(int(part) for part in entry.split("-"))
            for entry in str(point.get("completed_sets") or "").split(";")
            if entry and "-" in entry
        ]
        games = (int(point.get("games_1") or 0), int(point.get("games_2") or 0))
        in_tiebreak = games == (6, 6)
        prepared.append(
            {
                "clip": point["clip"],
                "set_lengths": completed,
                "games_completed_in_set": games[0] + games[1],
                "tiebreak_points": (
                    _tiebreak_point(point.get("points_1")) + _tiebreak_point(point.get("points_2"))
                    if in_tiebreak
                    else 0
                ),
                "completed_tiebreak_points": [
                    longest.get(index + 1, ASSUMED_TIEBREAK_POINTS)
                    for index in range(len(completed))
                ],
            }
        )
    return ends_parity(prepared)


def parity_orientation(parity: dict[str, int], anchor: str = "AB") -> dict[str, str]:
    """The orientation ends parity alone predicts, given the anchor point's orientation."""
    return {clip: (anchor[::-1] if value else anchor) for clip, value in sorted(parity.items())}


def constrain_by_parity(
    clusters: ClusterResult, parity: dict[str, int]
) -> tuple[dict[str, str], float]:
    """Force every point of one parity class onto one orientation, appearance deciding which.

    Two points played from the same ends must have the same near/far identity, so parity
    leaves exactly one free bit for the whole match. The bit is set by the appearance
    margins: each point votes for the orientation its own descriptor preferred, weighted by
    how strongly it preferred it. The returned confidence is the winning share of that
    weight, so a match where appearance and parity fight is not silently resolved.
    """
    weights = {0: 0.0, 1: 0.0}
    for clip, value in parity.items():
        orientation = clusters.orientation.get(clip)
        if orientation is None:
            continue
        weight = float(clusters.margin.get(clip, 0.0)) + 1e-6
        vote = 0 if (orientation == "AB") == (value == 0) else 1
        weights[vote] += weight
    total = weights[0] + weights[1]
    flip = weights[1] > weights[0]
    anchor = "BA" if flip else "AB"
    confidence = round(max(weights.values()) / total, 4) if total else 0.0
    return parity_orientation(parity, anchor), confidence


def standardised_depths(points: list[dict]) -> dict[str, dict[str, float]]:
    """clip -> per-side standard score of how deep behind their own baseline a player stood.

    ``near_depth``/``far_depth`` come in as any measure that grows with distance behind the
    player's own baseline (court metres when a homography exists, and the signed image
    row of the foot point when it does not). Near and far are standardised separately
    because the two are not on one scale, and over the match rather than over the point,
    so what is compared is one player's stance against the same player's usual stance.
    """
    values = {
        side: np.array(
            [float(p[f"{side}_depth"]) for p in points if p.get(f"{side}_depth") is not None],
            dtype=float,
        )
        for side in ("near", "far")
    }
    stats = {}
    for side, array in values.items():
        if len(array) < 2:
            stats[side] = (0.0, 1.0)
            continue
        spread = float(array.std())
        stats[side] = (float(array.mean()), spread if spread > 1e-9 else 1.0)
    out: dict[str, dict[str, float]] = {}
    for point in points:
        entry = {}
        for side in ("near", "far"):
            value = point.get(f"{side}_depth")
            if value is None:
                continue
            mean, spread = stats[side]
            entry[side] = round((float(value) - mean) / spread, 4)
        if len(entry) == 2:
            out[point["clip"]] = entry
    return out


def serving_sides(depths: dict[str, dict[str, float]]) -> dict[str, str]:
    """clip -> the side that served, read off the two standardised stance depths.

    The server has to stand within a stride of their own baseline; the receiver chooses,
    and at this level chooses to stand metres behind it. So of the two players the shallower
    one is serving. This is a per-point coin with a bias, not a per-point certainty; it is
    used only in aggregate, where a few hundred points make the bias decisive.
    """
    return {
        clip: ("near" if entry["near"] <= entry["far"] else "far") for clip, entry in depths.items()
    }


def anchor_from_scoreboard(rows: list[dict], *, board_names: dict[int, str] | None = None) -> dict:
    """Pin identity A to a scoreboard row, from who was serving on each point.

    Each row needs ``clip``, ``near_identity``, ``server_row`` (the row of the board the
    decoded score says is serving) and ``serving_side``. The identity playing on the
    serving side is the identity of that row; every point casts that vote and the majority
    wins. The confidence is the winning share, and the count of points on each side of the
    vote is reported so a caller can see how close it was.
    """
    votes = {"A": {1: 0, 2: 0}, "B": {1: 0, 2: 0}}
    used = 0
    for row in rows:
        near_identity = row.get("near_identity")
        server_row = row.get("server_row")
        side = row.get("serving_side")
        if near_identity is None or server_row not in (1, 2) or side not in ("near", "far"):
            continue
        far_identity = "B" if near_identity == "A" else "A"
        serving_identity = near_identity if side == "near" else far_identity
        votes[serving_identity][int(server_row)] += 1
        used += 1
    straight = votes["A"][1] + votes["B"][2]
    swapped = votes["A"][2] + votes["B"][1]
    total = straight + swapped
    mapping = {"A": 1, "B": 2} if straight >= swapped else {"A": 2, "B": 1}
    confidence = round(max(straight, swapped) / total, 4) if total else 0.0
    names = {}
    if board_names:
        names = {identity: board_names.get(row, "") for identity, row in mapping.items()}
    return {
        "identity_to_row": mapping,
        "names": names,
        "points_voting": used,
        "votes_for_mapping": max(straight, swapped),
        "votes_against_mapping": min(straight, swapped),
        "confidence": confidence,
    }


# --------------------------------------------------------------------------- resolution

MIN_PARITY_AGREEMENT = 0.75
MIN_ANCHOR_CONFIDENCE = 0.60
# Audit reels contain only a handful of points per match. Three independent serve-side
# observations are enough to make the scoreboard-row mapping a measurement rather than a
# roster-order convention; the confidence and parity witnesses still gate the result.
MIN_ANCHOR_POINTS = 3


def board_name_map(board_names: dict[int, str] | None, roster: Roster) -> dict[int, str]:
    """Scoreboard row -> the roster spelling of that row's surname.

    The reader returns what is printed on the board ("ALCARAZ"); the roster carries the
    name the rest of the pipeline uses ("carlos alcaraz"). A row is matched to a roster
    name when the board surname appears as one of its tokens, and otherwise keeps the
    board's own text, so an unrecognised board never silently borrows the wrong name.
    """
    if not board_names:
        return {}
    out: dict[int, str] = {}
    for row, printed in board_names.items():
        token = str(printed or "").strip().lower()
        chosen = token
        if token and roster.names:
            for name in roster.names:
                if token in name.split():
                    chosen = name
                    break
        out[int(row)] = chosen
    return out


def resolve_match(
    match_id: str,
    points: list[PointAppearance],
    *,
    games: list[dict] | None = None,
    scores: list[dict] | None = None,
    groups: dict[str, str] | None = None,
    depths: list[dict] | None = None,
    serve_sides: dict[str, str] | None = None,
    board_names: dict[int, str] | None = None,
    roster: Roster | None = None,
    anchor: dict[str, str] | None = None,
    provenance: dict | None = None,
    centre_sides: bool | None = None,
    min_parity_agreement: float = MIN_PARITY_AGREEMENT,
    min_anchor_confidence: float = MIN_ANCHOR_CONFIDENCE,
    min_anchor_points: int = MIN_ANCHOR_POINTS,
) -> dict:
    """The ``player_identity_v1.json`` payload for one match, from three witnesses.

    ``scores`` is one decoded ``score_grammar`` row per clip (``server``, ``set_number``,
    ``games_1``, ``games_2``, ``completed_sets``). It supplies both the ends parity and the
    scoreboard row that is serving. ``serve_sides`` gives the serve detector's direct
    near/far start-side evidence. It takes precedence over the stance estimate from
    ``depths`` on clips where it is available, with stance retained only as a fallback.
    When all three witnesses are present and agree, the names are a measurement
    (``scoreboard_server_anchor``); when they disagree the names are withheld rather than
    guessed.
    """
    roster = roster or roster_from_match_id(match_id)
    scores = scores or []
    by_clip = {row["clip"]: row for row in scores}
    # Per-set grouping is available for kit changes but is NOT derived from the score by
    # default: measured on the three finals it gains 2 points of agreement on one match and
    # loses 28 on another, so a caller has to ask for it. Pass ``groups`` to switch it on.
    clusters = cluster_two_identities(points, groups=groups)

    parity = ends_parity(games) if games else (parity_from_scores(scores) if scores else {})
    # Removing the per-side mean cancels the near/far bias only when both players have
    # been seen on both sides; on a corpus whose points are all played from one end the
    # near mean IS one player's mean and subtracting it throws the identity away with the
    # bias. So it is on exactly when the ends actually change.
    if centre_sides is None:
        centre_sides = len(set(parity.values())) > 1
    if centre_sides:
        clusters = cluster_two_identities(remove_side_bias(points), groups=groups)
    appearance_orientation = dict(clusters.orientation)
    anchor_orientation = None
    if appearance_orientation:
        anchor_orientation = appearance_orientation[sorted(appearance_orientation)[0]]

    orientation = dict(appearance_orientation)
    parity_confidence = None
    parity_expected: dict[str, str] = {}
    agree = disagree = 0
    if parity and anchor_orientation is not None:
        constrained, parity_confidence = constrain_by_parity(clusters, parity)
        parity_expected = parity_orientation(parity, anchor_orientation)
        for clip, expected in parity_expected.items():
            if clip in appearance_orientation:
                agreed = appearance_orientation[clip] == expected
                agree += int(agreed)
                disagree += int(not agreed)
        orientation = {**orientation, **constrained}
    agreement = round(agree / (agree + disagree), 4) if (agree + disagree) else None

    depth_scores = standardised_depths(depths) if depths else {}
    stance_sides = serving_sides(depth_scores)
    direct_sides = {
        clip: side for clip, side in (serve_sides or {}).items() if side in ("near", "far")
    }
    sides = {**stance_sides, **direct_sides}
    side_sources = {
        clip: ("serve_start" if clip in direct_sides else "stance_depth") for clip in sides
    }
    anchor_rows = [
        {
            "clip": clip,
            "near_identity": orientation[clip][0],
            "server_row": int(by_clip[clip]["server"])
            if str(by_clip.get(clip, {}).get("server", "")).strip() in ("1", "2")
            else None,
            "serving_side": side,
        }
        for clip, side in sides.items()
        if clip in orientation and clip in by_clip
    ]
    printed = board_name_map(board_names, roster)
    anchor_report = anchor_from_scoreboard(anchor_rows, board_names=printed)

    witness_failures: list[str] = []
    if parity and agreement is not None and agreement < min_parity_agreement:
        witness_failures.append(
            f"appearance agrees with ends parity on {agreement:.4f} of points, "
            f"below {min_parity_agreement}"
        )
    anchored = bool(printed) and anchor_report["points_voting"] >= min_anchor_points
    if anchored and anchor_report["confidence"] < min_anchor_confidence:
        witness_failures.append(
            f"scoreboard server anchor confidence {anchor_report['confidence']:.4f}, "
            f"below {min_anchor_confidence}"
        )
        anchored = False

    names: dict[str, str] = {}
    name_source = "none"
    if anchor:
        names = {"A": anchor.get("A", ""), "B": anchor.get("B", "")}
        name_source = "explicit_anchor"
    elif anchored and all(anchor_report["names"].get(key) for key in IDENTITIES):
        names = dict(anchor_report["names"])
        name_source = "scoreboard_server_anchor"
    elif roster.names:
        names = {"A": roster.names[0], "B": roster.names[1]}
        name_source = "roster_order_unanchored"
    if witness_failures and name_source != "explicit_anchor":
        # Fail closed: an absent name is recoverable downstream, a wrong one is not.
        names = {}
        name_source = "withheld_witness_disagreement"

    confidence = 0.0
    if name_source == "scoreboard_server_anchor":
        confidence = round(
            anchor_report["confidence"] * (parity_confidence or 1.0) * (agreement or 1.0), 4
        )
    elif name_source == "explicit_anchor":
        confidence = 1.0

    rows = []
    for point in sorted(points, key=lambda p: p.clip):
        chosen = orientation.get(point.clip)
        near_identity = chosen[0] if chosen else None
        far_identity = chosen[1] if chosen else None
        appearance = appearance_orientation.get(point.clip)
        expected = parity_expected.get(point.clip)
        rows.append(
            {
                "clip": point.clip,
                "near_identity": near_identity,
                "far_identity": far_identity,
                "near_name": names.get(near_identity, "") if near_identity else "",
                "far_name": names.get(far_identity, "") if far_identity else "",
                "near_frames": point.near_frames,
                "far_frames": point.far_frames,
                "appearance_orientation": appearance,
                "appearance_margin": clusters.margin.get(point.clip),
                "cluster_group": clusters.groups.get(point.clip),
                "parity": parity.get(point.clip),
                "parity_expected_orientation": expected,
                "parity_agrees": (
                    None if expected is None or appearance is None else appearance == expected
                ),
                "serving_side": sides.get(point.clip),
                "serving_side_source": side_sources.get(point.clip),
                "server_row": (
                    int(by_clip[point.clip]["server"])
                    if str(by_clip.get(point.clip, {}).get("server", "")).strip() in ("1", "2")
                    else None
                ),
            }
        )

    return {
        "schema": SCHEMA,
        "match_id": match_id,
        "identities": [
            {
                "identity": key,
                "name": names.get(key, ""),
                "name_source": name_source,
                "name_confidence": confidence,
                "scoreboard_row": anchor_report["identity_to_row"].get(key)
                if name_source == "scoreboard_server_anchor"
                else None,
                "points": sum(
                    1 for row in rows if key in (row["near_identity"], row["far_identity"])
                ),
                "frames": sum(
                    (row["near_frames"] if row["near_identity"] == key else 0)
                    + (row["far_frames"] if row["far_identity"] == key else 0)
                    for row in rows
                ),
            }
            for key in IDENTITIES
        ],
        "points": rows,
        "appearance": {
            "points_clustered": clusters.points_used,
            "mean_within_identity_distance": clusters.within,
            "mean_between_identity_distance": clusters.between,
            "separation": (
                round(clusters.between - clusters.within, 4)
                if np.isfinite(clusters.within) and np.isfinite(clusters.between)
                else None
            ),
            "cluster_groups": len(set(clusters.groups.values())) if clusters.groups else 0,
            "side_bias_removed": centre_sides,
            "group_link_margin": clusters.group_link_margin,
        },
        "parity": {
            "available": bool(parity),
            "points_with_game_index": len(parity),
            "anchor_clip": sorted(appearance_orientation)[0] if appearance_orientation else None,
            "agree": agree,
            "disagree": disagree,
            "agreement": agreement,
            "constrained": bool(parity and anchor_orientation is not None),
            "constraint_confidence": parity_confidence,
        },
        "server_anchor": {
            **anchor_report,
            "side_source_counts": {
                source: sum(side_sources.get(row["clip"]) == source for row in anchor_rows)
                for source in ("serve_start", "stance_depth")
            },
        },
        "witnesses": {
            "appearance": bool(appearance_orientation),
            "ends_parity": bool(parity),
            "scoreboard_server": anchored,
            "agree": not witness_failures,
            "failures": witness_failures,
        },
        "roster": {
            "names": list(roster.names) if roster.names else None,
            "source": roster.source,
            "confidence": roster.confidence,
            "anchored": name_source in ("explicit_anchor", "scoreboard_server_anchor"),
        },
        "provenance": provenance or {},
    }


def names_are_safe(payload: dict) -> bool:
    """True when the payload's names may be written onto a track row.

    The end-to-end runner calls this before stamping names. A parsed roster is useful
    context, but its order is not evidence that identity A is the first name: only an
    explicit anchor or the agreeing scoreboard/server witnesses may cross that boundary.
    """
    witnesses = payload.get("witnesses") or {}
    if witnesses and not witnesses.get("agree", True):
        return False
    identities = payload.get("identities") or []
    sources = {entry.get("name_source") for entry in identities}
    safe_sources = {"explicit_anchor", "scoreboard_server_anchor"}
    return (
        bool(identities)
        and sources <= safe_sources
        and all(entry.get("name") for entry in identities)
    )


def write_identity(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def names_by_clip(payload: dict) -> dict[str, dict[str, str]]:
    """clip -> {side: name}, the shape ``player_side_association.track_rows`` wants.

    Empty unless the identities have a measured anchor and the witnesses agree. Every
    consumer of the artifact goes through here, so an unanchored roster convention or a
    witness disagreement emits no name rather than one that is plausibly reversed.
    """
    if not names_are_safe(payload):
        return {}
    return {
        row["clip"]: {"near": row["near_name"], "far": row["far_name"]}
        for row in payload.get("points", [])
    }
