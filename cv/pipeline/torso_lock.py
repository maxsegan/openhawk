"""Drop ball candidates that stay inside a player's torso.

A ball crosses a body for a frame or two. The Halle lock sits on the neon
shirt for the whole rally, and the real ball is the other blob on those
frames. A candidate is refused only when it can be chained, inside one
torso, for ``min_run`` frames. The other blobs are left for the tracker to
reacquire. Off, the caller does not call this and the pool is unchanged.
"""

from __future__ import annotations

import csv
from pathlib import Path

from cv.pipeline.player_identity import torso_box

POLICY_KEY = "torso_lock_rejection"
MIN_RUN = 6
LINK_PX = 24.0
# The run was first counted in pooled candidate rows, so six detectors seeing a ball cross
# the body for two frames made a "run" of twelve. Counting distinct frames matches the
# intent (a ball crosses a body for a few frames; a shirt lock persists).
RUN_IN_FRAMES = False
# A far player's torso is small, so a real ball arriving at or leaving his racket stays
# inside it for six frames and the lock deleted it (ceiling audit balltrack T-a: 39 real
# flight rows on panel E). With KEEP_JOINED, a chain in a torso shorter than
# JOINED_TORSO_MAX_PX is kept when BOTH ends join a candidate outside every torso within
# JOIN_FRAMES frames, at most JOIN_PX_PER_FRAME per frame of gap from the chain's end: the
# ball enters and leaves the body at ball speed. A shirt lock sits in a near player's
# large torso and does not. Development chains (streak_centre/CHAIN_FEATURES.json): keeps
# 21 of 40 real flight chains and none of the Halle chains off the ball. Default on 2026-09-29 (tracking_composition).
KEEP_JOINED = True
JOINED_TORSO_MAX_PX = 75.0
JOIN_FRAMES = 3
JOIN_PX_PER_FRAME = 12.0


def _inside(point: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
    x, y = point
    x0, y0, x1, y1 = box
    return x0 <= x <= x1 and y0 <= y <= y1


def _torso(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return torso_box(box)


def torso_chains(
    frames: list[int],
    points: list[tuple[float, float]],
    boxes_by_frame: dict[int, list[tuple[float, float, float, float]]],
    *,
    link_px: float = LINK_PX,
) -> list[tuple[list[int], list[tuple[float, float, float, float]]]]:
    """Chains of candidates linked frame to frame inside one player's torso.

    Returns, per chain, the member indices in frame order and the torso box of
    each member. Chains do not jump to a different player.
    """

    if link_px < 0:
        raise ValueError("link radius must be non-negative")
    if len(frames) != len(points):
        raise ValueError("frames and points must align")
    count = len(points)
    owners: list[int | None] = [None] * count
    torsos: list[tuple[float, float, float, float] | None] = [None] * count
    for index, (frame, point) in enumerate(zip(frames, points, strict=True)):
        containing = [
            (position, _torso(box))
            for position, box in enumerate(boxes_by_frame.get(int(frame), ()))
            if _inside(point, _torso(box))
        ]
        if len(containing) == 1:
            owners[index], torsos[index] = containing[0]
    by_frame: dict[int, list[int]] = {}
    for index, frame in enumerate(frames):
        by_frame.setdefault(int(frame), []).append(index)
    ordered = sorted(by_frame)
    parent = list(range(count))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    position = {frame: order for order, frame in enumerate(ordered)}
    for frame, members in by_frame.items():
        nxt = position[frame] + 1
        if nxt >= len(ordered) or ordered[nxt] != frame + 1:
            continue
        for left in members:
            if owners[left] is None:
                continue
            x, y = points[left]
            for right in by_frame[ordered[nxt]]:
                if owners[right] != owners[left]:
                    continue
                ox, oy = points[right]
                if (ox - x) ** 2 + (oy - y) ** 2 <= link_px**2:
                    union(left, right)
    groups: dict[int, list[int]] = {}
    for index in range(count):
        if owners[index] is not None:
            groups.setdefault(find(index), []).append(index)
    chains = []
    for members in groups.values():
        members.sort(key=lambda index: (frames[index], index))
        chains.append((members, [torsos[index] for index in members]))
    return chains


def relative_travel(
    points: list[tuple[float, float]], torsos: list[tuple[float, float, float, float]]
) -> float:
    """How far a chain moves across its torso, in torso heights.

    Positions are taken relative to the torso centre, so a shirt carried by a
    running player stays put while a ball crossing the body travels.
    """

    rel = [
        (
            (x - (b[0] + b[2]) / 2) / max(b[3] - b[1], 1.0),
            (y - (b[1] + b[3]) / 2) / max(b[3] - b[1], 1.0),
        )
        for (x, y), b in zip(points, torsos, strict=True)
    ]
    return max(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 for ax, ay in rel for bx, by in rel)


def _joined_both_ends(
    members: list[int],
    frames: list[int],
    points: list[tuple[float, float]],
    outside: dict[int, list[tuple[float, float]]],
) -> bool:
    """Both chain ends meet an outside-torso candidate at ball speed within JOIN_FRAMES."""

    by_frame: dict[int, list[tuple[float, float]]] = {}
    for index in members:
        by_frame.setdefault(int(frames[index]), []).append(points[index])

    def end(frame: int, direction: int) -> bool:
        xs, ys = zip(*by_frame[frame])
        x, y = sum(xs) / len(xs), sum(ys) / len(ys)
        for gap in range(1, JOIN_FRAMES + 1):
            reach = (JOIN_PX_PER_FRAME * gap) ** 2
            if any(
                (ox - x) ** 2 + (oy - y) ** 2 <= reach
                for ox, oy in outside.get(frame + direction * gap, ())
            ):
                return True
        return False

    return end(min(by_frame), -1) and end(max(by_frame), 1)


def persistent_torso_mask(
    frames: list[int],
    points: list[tuple[float, float]],
    boxes_by_frame: dict[int, list[tuple[float, float, float, float]]],
    *,
    min_run: int = MIN_RUN,
    link_px: float = LINK_PX,
    max_relative_travel: float | None = None,
    run_in_frames: bool | None = None,
    keep_joined: bool | None = None,
) -> list[bool]:
    """True where a candidate belongs to a torso chain of at least ``min_run``.

    ``frames[i]`` is the frame of ``points[i]``. Boxes are whole-person boxes
    in the same pixel space as the points; the chain uses the torso, the upper
    60 percent. With ``max_relative_travel``, a chain that moves farther than
    that across its torso (a ball crossing the body) is kept. ``run_in_frames``
    (default ``RUN_IN_FRAMES``) counts the run in distinct frames, not rows.
    ``keep_joined`` (default ``KEEP_JOINED``) keeps a small-torso chain whose two ends
    join outside-torso candidates at ball speed.
    """

    if run_in_frames is None:
        run_in_frames = RUN_IN_FRAMES
    if keep_joined is None:
        keep_joined = KEEP_JOINED
    if min_run < 2:
        raise ValueError("a persistent lock is at least two frames")
    mask = [False] * len(points)
    outside: dict[int, list[tuple[float, float]]] = {}
    if keep_joined:
        for frame, point in zip(frames, points, strict=True):
            if not any(_inside(point, _torso(box)) for box in boxes_by_frame.get(int(frame), ())):
                outside.setdefault(int(frame), []).append(point)
    for members, torsos in torso_chains(frames, points, boxes_by_frame, link_px=link_px):
        run = len({int(frames[index]) for index in members}) if run_in_frames else len(members)
        if run < min_run:
            continue
        if (
            keep_joined
            and sum(b[3] - b[1] for b in torsos) / len(torsos) < JOINED_TORSO_MAX_PX
            and _joined_both_ends(members, frames, points, outside)
        ):
            continue
        if max_relative_travel is not None and (
            relative_travel([points[index] for index in members], torsos) > max_relative_travel
        ):
            continue
        for index in members:
            mask[index] = True
    return mask


def load_torso_boxes(path: Path) -> dict[str, dict[int, list[tuple[float, float, float, float]]]]:
    """Sided person boxes, keyed by clip then native frame.

    The torso is derived later. A file without native columns is refused rather
    than read as a different pixel space.
    """

    boxes: dict[str, dict[int, list[tuple[float, float, float, float]]]] = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"clip", "frame", "x0_native", "y0_native", "x1_native", "y1_native"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(f"{path} is missing native player-box columns")
        for row in reader:
            frame = int("".join(ch for ch in str(row["frame"]) if ch.isdigit()))
            box = (
                float(row["x0_native"]),
                float(row["y0_native"]),
                float(row["x1_native"]),
                float(row["y1_native"]),
            )
            boxes.setdefault(str(row["clip"]), {}).setdefault(frame, []).append(box)
    return boxes


def without_torso_locks(
    frame_observations: dict,
    boxes_by_clip_frame: dict,
    *,
    min_run: int = MIN_RUN,
    run_in_frames: bool | None = None,
    keep_joined: bool | None = None,
):
    """Return the pool with persistent torso candidates removed.

    ``frame_observations`` maps frame to objects with ``x`` and ``y``. The
    returned lists are new; the input objects are not modified. A frame whose
    every candidate was a lock becomes an empty list, which is a miss the
    tracker already knows how to cross.
    """

    frames: list[int] = []
    points: list[tuple[float, float]] = []
    flat: list[tuple[int, int]] = []
    for frame, observations in frame_observations.items():
        for position, observation in enumerate(observations):
            frames.append(int(frame))
            points.append((float(observation.x), float(observation.y)))
            flat.append((int(frame), position))
    mask = persistent_torso_mask(
        frames,
        points,
        boxes_by_clip_frame,
        min_run=min_run,
        run_in_frames=run_in_frames,
        keep_joined=keep_joined,
    )
    drop = {flat[index] for index, locked in enumerate(mask) if locked}
    return {
        frame: [
            observation
            for position, observation in enumerate(observations)
            if (int(frame), position) not in drop
        ]
        for frame, observations in frame_observations.items()
    }


def _frame_number(value: str) -> int:
    return int("".join(ch for ch in str(value) if ch.isdigit()))


def filter_candidate_streams(
    match_dir: Path,
    names: list[str],
    boxes_path: Path,
    output: Path,
    *,
    min_run: int = MIN_RUN,
    run_in_frames: bool | None = None,
    keep_joined: bool | None = None,
) -> dict[str, int]:
    """Write every named candidate stream to ``output`` without its torso chains.

    The mask is computed on the pooled candidates of each clip, so a shirt that
    one detector sees on some frames and another on the rest is still one chain.
    Coordinate sidecars are copied unchanged. Returns rows dropped per stream.
    """

    boxes = load_torso_boxes(boxes_path)
    tables: dict[str, tuple[list[str], list[dict]]] = {}
    for name in names:
        with (Path(match_dir) / name).open(newline="") as handle:
            reader = csv.DictReader(handle)
            tables[name] = (list(reader.fieldnames or []), list(reader))
    by_clip: dict[str, list[tuple[str, int]]] = {}
    for name, (_, rows) in tables.items():
        for index, row in enumerate(rows):
            by_clip.setdefault(row["clip"], []).append((name, index))
    drop: set[tuple[str, int]] = set()
    for clip, members in by_clip.items():
        rows = [tables[name][1][index] for name, index in members]
        mask = persistent_torso_mask(
            [_frame_number(row["frame"]) for row in rows],
            [(float(row["x_native"]), float(row["y_native"])) for row in rows],
            boxes.get(clip, {}),
            min_run=min_run,
            run_in_frames=run_in_frames,
            keep_joined=keep_joined,
        )
        drop.update(member for member, locked in zip(members, mask, strict=True) if locked)
    output.mkdir(parents=True, exist_ok=True)
    dropped = {}
    for name, (fields, rows) in tables.items():
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(row for index, row in enumerate(rows) if (name, index) not in drop)
        sidecar = Path(match_dir) / f"{name}.coordinates.json"
        if sidecar.is_file():
            (output / sidecar.name).write_bytes(sidecar.read_bytes())
        dropped[name] = sum(1 for member in drop if member[0] == name)
    return dropped


# The stream filter also drops a real ball that crosses a near player's large torso at a
# bounce or contact (combined-v3: fresh E019 148-152, dev c05 415-422, e03 582-587), and
# chain features do not tell those chains from a racket or shirt (streak_centre
# CHAIN_FEATURES.json). With GAP_RESTORE the tracker also runs on the unfiltered pool, and a
# run of its rows is copied into the torso-lock track only where the torso-lock track has no
# row at all and both tracks agree, within GAP_AGREE_PX, on the frame just before and just
# after the run. The torso-lock track is never overwritten, so a shirt it replaced with the
# real ball stays replaced. A run touching a torso of JOINED_TORSO_MAX_PX or more (a near
# player) must also enter and leave on the unfiltered track's own velocity: the step into the
# run and the step out of it may differ from the neighbouring step outside by at most
# GAP_JUMP_PX. There the unfiltered track can jump onto hands or racket while the ball is
# hidden (E019 148-151, viewed: 52 px jump); the real ball at a near bounce or contact
# continues its path (e03 606-611, D A1_C02 245-252, dev c05 415-422, viewed). Default on 2026-09-29.
GAP_RESTORE = True
GAP_AGREE_PX = 8.0
GAP_MAX_FRAMES = 15
GAP_JUMP_PX = 15.0


def restore_lock_gaps(
    lock_rows: list[dict],
    free_rows: list[dict],
    *,
    agree_px: float = GAP_AGREE_PX,
    max_frames: int = GAP_MAX_FRAMES,
    boxes_by_clip: dict | None = None,
    jump_px: float = GAP_JUMP_PX,
) -> tuple[list[dict], int]:
    """The torso-lock track with the unfiltered track's rows in its bracketed gaps.

    Rows are track CSV dicts with ``clip``, ``frame``, ``x_native`` and ``y_native``. A gap
    is a run of consecutive frames, at most ``max_frames`` long, on which the unfiltered
    track has a row and the torso-lock track has none; the frames on either side must hold a
    row in both tracks within ``agree_px``. With ``boxes_by_clip`` (``load_torso_boxes``) a
    run touching a torso at least ``JOINED_TORSO_MAX_PX`` tall is skipped. Returns the rows
    in (clip, frame) order and the number restored.
    """

    def key(row: dict) -> tuple[str, int]:
        return str(row["clip"]), _frame_number(row["frame"])

    def xy(row: dict) -> tuple[float, float]:
        return float(row["x_native"]), float(row["y_native"])

    lock = {key(row): row for row in lock_rows}
    free = {key(row): row for row in free_rows}

    def agree(k: tuple[str, int]) -> bool:
        if k not in lock or k not in free:
            return False
        (ax, ay), (bx, by) = xy(lock[k]), xy(free[k])
        return (ax - bx) ** 2 + (ay - by) ** 2 <= agree_px**2

    def near_player(k: tuple[str, int]) -> bool:
        if boxes_by_clip is None:
            return False
        point = xy(free[k])
        return any(
            _inside(point, torso) and torso[3] - torso[1] >= JOINED_TORSO_MAX_PX
            for torso in map(_torso, boxes_by_clip.get(k[0], {}).get(k[1], ()))
        )

    def step(clip: str, frame: int) -> tuple[float, float] | None:
        a, b = free.get((clip, frame)), free.get((clip, frame + 1))
        if a is None or b is None:
            return None
        (ax, ay), (bx, by) = xy(a), xy(b)
        return bx - ax, by - ay

    def continuous(clip: str, first: int, last: int) -> bool:
        for inner, outer in ((first - 1, first - 2), (last, last + 1)):
            a, b = step(clip, inner), step(clip, outer)
            if a is None or b is None:
                return False
            if (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 > jump_px**2:
                return False
        return True

    restored: dict[tuple[str, int], dict] = {}
    missing = sorted(k for k in free if k not in lock)
    index = 0
    while index < len(missing):
        clip, first = missing[index]
        last = first
        while index + 1 < len(missing) and missing[index + 1] == (clip, last + 1):
            index += 1
            last += 1
        index += 1
        run = [(clip, frame) for frame in range(first, last + 1)]
        if (
            len(run) <= max_frames
            and agree((clip, first - 1))
            and agree((clip, last + 1))
            and (not any(near_player(k) for k in run) or continuous(clip, first, last))
        ):
            for frame in range(first, last + 1):
                restored[(clip, frame)] = free[(clip, frame)]
    merged = {**lock, **restored}
    return [merged[k] for k in sorted(merged)], len(restored)


def restore_lock_gaps_file(
    lock_path: Path, free_path: Path, output: Path, boxes_path: Path | None = None
) -> int:
    """File form of ``restore_lock_gaps``; keeps the torso-lock track's columns and sidecar."""

    with Path(lock_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        lock_rows = list(reader)
    with Path(free_path).open(newline="") as handle:
        free_rows = list(csv.DictReader(handle))
    boxes = load_torso_boxes(boxes_path) if boxes_path is not None else None
    rows, count = restore_lock_gaps(lock_rows, free_rows, boxes_by_clip=boxes)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    sidecar = Path(f"{lock_path}.coordinates.json")
    if sidecar.is_file() and Path(lock_path) != output:
        Path(f"{output}.coordinates.json").write_bytes(sidecar.read_bytes())
    return count
