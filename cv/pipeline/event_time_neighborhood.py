"""The local timing neighborhood of one emitted event, as source evidence.

An emission row already carries its own row of the model's timing softmax
(``event_time_distribution``).  That answers "where inside its crop window does
this one row think the event was", but S6 also needs to propose a *timing*
separately from the occurrence it already decoded: the neighbouring crop rows
scored the same physical moment from their own windows, and their softmaxes are
the evidence for that.

This module collects those neighbours and nothing else:

* the anchor is the event's **original emitted candidate frame** -- the native
  crop row the decoder fired on -- never a corrected or fitted epoch;
* neighbours are the **available** rows of the **same clip** within
  ``RADIUS_NATIVE_FRAMES`` of that anchor.  Missing rows stay missing: no row is
  interpolated, no clip boundary is crossed, and a full integer native cadence
  is the only case that yields all ``2 * RADIUS + 1`` rows;
* every neighbour keeps its own ``source_row_index`` into the prediction
  artifact, its four-class scores and its own verbatim timing record, whose
  offsets remain relative to **that neighbour's** candidate frame.

Classifier-``none`` rows are kept: timing supervision targets the nearest event
also on neighboring none-class rows. Their PMFs can locate an event without
establishing its type or occurrence.

The block is strictly additive: without a persisted distribution there is no
neighborhood, and the emission dictionaries are the ones the old producer wrote.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from cv.pipeline import event_time_distribution as timing

SCHEMA = "event_time_neighborhood_v1"
SOURCE = timing.SOURCE
STATUS = timing.AVAILABLE
KEY = "time_neighborhood"
# The neighbourhood half-width in native frames.  It matches the model's own
# time-head radius: a row cannot speak about a frame outside its crop window.
RADIUS_NATIVE_FRAMES = timing.TIME_RADIUS


def build_index(
    clips: Sequence, frames: Sequence, *, allow_ambiguous: bool = False
) -> dict[str, tuple[np.ndarray, np.ndarray] | None]:
    """Return ``clip -> (sorted integer frames, their prediction row indices)``.

    Built once per emission pass so a neighborhood lookup is a binary search
    rather than a scan of every prediction row.  Native crop frames must be
    finite integers and unique inside a clip; anything else would make "the row
    at frame f" ambiguous, which is not something to resolve by guessing.
    """

    positions: dict[str, list[int]] = defaultdict(list)
    for position, clip in enumerate(np.asarray(clips).tolist()):
        positions[str(clip)].append(position)
    values = np.asarray(frames, dtype=np.float64)
    if len(values) != len(np.asarray(clips)):
        raise ValueError("native crop frames must be row-aligned with clips")
    index: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
    for clip, rows in positions.items():
        selected = values[rows]
        if not np.isfinite(selected).all() or np.any(selected != np.floor(selected)):
            raise ValueError("native crop frames must be finite integers")
        order = np.argsort(selected, kind="stable")
        ordered = selected[order].astype(np.int64)
        if np.any(np.diff(ordered) <= 0):
            if allow_ambiguous:
                index[clip] = None
                continue
            raise ValueError(f"native crop frames repeat inside clip {clip}")
        index[clip] = (ordered, np.asarray(rows, dtype=np.int64)[order])
    return index


def _class_scores(values: np.ndarray, classes: Sequence[str]) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if (
        values.shape != (len(classes),)
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("finite normalised four-class prediction scores required")
    return dict(zip(classes, values.tolist(), strict=True))


def record(
    row: int,
    *,
    index: Mapping[str, tuple[np.ndarray, np.ndarray]],
    clip: str,
    candidate_frame: float,
    probabilities: np.ndarray,
    classes: Sequence[str],
    pmf: np.ndarray,
    grid: np.ndarray,
) -> dict:
    """Return the neighborhood block for one emitted row's own source row.

    ``pmf``/``grid`` must already have passed ``event_time_distribution``.
    """

    ordered, sources = index[clip]
    anchor = float(candidate_frame)
    if anchor != float(int(anchor)):
        raise ValueError("neighborhood anchor must be a native integer frame")
    anchor = int(anchor)
    low = int(np.searchsorted(ordered, anchor - RADIUS_NATIVE_FRAMES, side="left"))
    high = int(np.searchsorted(ordered, anchor + RADIUS_NATIVE_FRAMES, side="right"))
    rows = [
        {
            "source_row_index": int(source),
            "candidate_frame": float(frame),
            "class_probabilities": _class_scores(probabilities[int(source)], classes),
            "time_distribution": timing.record(pmf[int(source)], grid[int(source)]),
        }
        for frame, source in zip(ordered[low:high], sources[low:high], strict=True)
    ]
    if not any(neighbour["source_row_index"] == int(row) for neighbour in rows):
        raise ValueError("neighborhood does not contain the emitted row it anchors")
    return {
        "schema": SCHEMA,
        "source": SOURCE,
        "status": STATUS,
        "calibrated": False,
        "candidate_frame_anchor": float(anchor),
        "radius_native_frames": int(RADIUS_NATIVE_FRAMES),
        "rows": rows,
    }


def validate(
    block,
    *,
    clip: str,
    candidate_frame: float,
    classes: Sequence[str],
    clips: Sequence | None = None,
    frames: Sequence | None = None,
) -> None:
    """Refuse a neighborhood a producer could not honestly have written.

    ``clips``/``frames`` are the prediction artifact's row-aligned arrays; when
    they are supplied each neighbour's ``source_row_index`` is checked to be the
    clip's own row at the declared frame, which is what rejects a neighborhood
    stitched across clips or re-pointed at another row.
    """

    if not isinstance(block, Mapping):
        raise ValueError("time neighborhood must be a mapping")
    if (
        block.get("schema") != SCHEMA
        or block.get("source") != SOURCE
        or block.get("status") != STATUS
        or block.get("calibrated") is not False
        or block.get("radius_native_frames") != RADIUS_NATIVE_FRAMES
    ):
        raise ValueError("time neighborhood declares a different contract")
    anchor = block.get("candidate_frame_anchor")
    if not isinstance(anchor, float) or anchor != float(candidate_frame):
        raise ValueError("time neighborhood is not anchored on its row's candidate frame")
    rows = block.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("time neighborhood needs its available source rows")
    if len(rows) > 2 * RADIUS_NATIVE_FRAMES + 1:
        raise ValueError("time neighborhood carries more rows than its radius allows")
    previous = None
    anchored = False
    for neighbour in rows:
        if not isinstance(neighbour, Mapping):
            raise ValueError("time neighborhood row must be a mapping")
        frame = neighbour.get("candidate_frame")
        source = neighbour.get("source_row_index")
        if not isinstance(frame, float) or not np.isfinite(frame) or frame != float(int(frame)):
            raise ValueError("time neighborhood frames must be native integers")
        if type(source) is not int or source < 0:
            raise ValueError("time neighborhood row needs its prediction row index")
        if previous is not None and frame <= previous:
            raise ValueError("time neighborhood frames must be unique and ordered")
        previous = frame
        if abs(frame - anchor) > RADIUS_NATIVE_FRAMES:
            raise ValueError("time neighborhood row lies outside its declared radius")
        anchored = anchored or frame == anchor
        scores = neighbour.get("class_probabilities")
        if not isinstance(scores, Mapping) or set(scores) != set(classes):
            raise ValueError("time neighborhood row needs one score per event class")
        if not all(isinstance(scores[name], float) for name in classes):
            raise ValueError("time neighborhood class scores must be numbers")
        _class_scores(np.asarray([scores[name] for name in classes], dtype=np.float64), classes)
        distribution = neighbour.get("time_distribution")
        if not isinstance(distribution, Mapping):
            raise ValueError("time neighborhood row needs its own timing distribution")
        if (
            distribution.get("schema") != timing.SCHEMA
            or distribution.get("source") != timing.SOURCE
            or distribution.get("status") != timing.AVAILABLE
            or distribution.get("calibrated") is not False
            or distribution.get("offset_reference") != timing.OFFSET_REFERENCE
        ):
            raise ValueError("time neighborhood row declares a different timing contract")
        scored, offsets = distribution.get("probabilities"), distribution.get("offset_frames")
        if not all(
            isinstance(values, list)
            and len(values) == timing.TIME_BINS
            and all(isinstance(value, float) for value in values)
            for values in (scored, offsets)
        ):
            raise ValueError(f"time neighborhood row needs {timing.TIME_BINS} timing bins")
        timing.validated(
            np.asarray([scored], dtype=np.float64),
            np.asarray([offsets], dtype=np.float64),
            rows=1,
        )
        if clips is not None and frames is not None:
            if source >= len(np.asarray(frames)):
                raise ValueError("time neighborhood row index is outside the prediction artifact")
            if str(np.asarray(clips)[source]) != str(clip):
                raise ValueError("time neighborhood crosses a clip boundary")
            if float(np.asarray(frames)[source]) != frame:
                raise ValueError("time neighborhood row index is not the declared native frame")
    if not anchored:
        raise ValueError("time neighborhood omits its own anchor row")


__all__ = [
    "KEY",
    "RADIUS_NATIVE_FRAMES",
    "SCHEMA",
    "SOURCE",
    "STATUS",
    "build_index",
    "record",
    "validate",
]
