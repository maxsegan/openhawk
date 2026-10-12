"""Source-only flight coverage inventory shared by preparation and scope readers.

This is a pure extraction of the ownership rules ``prepare_attempt`` already
applies: one matching camera document, supported-camera visible labels and the
fixed train/check partition.  It reads supplied events, ball labels and cameras
only.  No fitted state, candidate, gate or reference file is consulted, and
nothing here decides or derives a physical ending.
"""

from __future__ import annotations

import numpy as np

#: Unchanged fixed coverage requirement per flight.
TRAIN_MINIMUM = 4
CHECK_MINIMUM = 1


def visible_inputs(
    attempt: dict,
    cameras_document: dict,
    *,
    observation_fallback: bool = False,
    fallback_receipt: list[dict] | None = None,
) -> tuple[dict[int, dict], dict[int, dict]]:
    """Exactly ``prepare_attempt``'s camera identity and visible-label admission.

    Returns every camera row keyed by native frame and the visible ball labels
    that own a supported matching camera.  Under the declared fallback an
    unsupported visible front is omitted, which is the treatment a non-visible
    exposure already receives; it never implies termination.
    """
    camera_rows = cameras_document["cameras"]
    cameras = {int(row["frame"]): row for row in camera_rows}
    labels = {
        int(row["frame"]): row for row in attempt["owner_ball_labels"] if row["status"] == "visible"
    }
    if (
        len(cameras) != len(camera_rows)
        or cameras_document["clip"] != attempt["point_clip"]
        or cameras_document["match_id"] != attempt["match_id"]
    ):
        raise ValueError("one matching camera document per attempt required")
    unsupported = sorted(
        frame for frame in labels if frame not in cameras or cameras[frame]["status"] != "supported"
    )
    if unsupported:
        if not observation_fallback:
            raise ValueError("every visible label requires one supported matching camera")
        for frame in unsupported:
            del labels[frame]
        if fallback_receipt is not None:
            fallback_receipt.append(
                {
                    "fallback": "unsupported_camera_observations_omitted",
                    "frames": unsupported,
                    "retained_visible_observations": len(labels),
                }
            )
    return cameras, labels


def frame_partition(
    labels,
    start: float,
    end: float,
    *,
    inclusive_end: bool,
    observation_partition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One flight's native/train/check rows under the existing partition rules.

    ``inclusive_end`` is the original terminal-flight rule; an interior boundary
    stays half open, so a row at the closing contact belongs to the next flight.
    """
    from cv.experiments.connected_shooting import observation_partition as partition

    partition.validate(observation_partition)
    frames = np.asarray(
        sorted(
            frame
            for frame in labels
            if start <= frame and (frame < end or inclusive_end and frame == end)
        ),
        float,
    )
    train = frames if observation_partition == "all_native" else frames[frames % 5 != 0]
    check = frames[frames % 5 == 0]
    if observation_partition == "all_native" and not len(check):
        # A diagnostic copy of actual fitted observations, never fabricated
        # check evidence or a requirement to withhold one exposure.
        check = frames
    return frames, train, check


def domain(
    attempt: dict, events: list[dict] | None = None, end_frame: float | None = None
) -> tuple[list[dict], list[float]]:
    """Original source flight domain: supplied contacts plus the declared endpoint.

    The endpoint is whatever the source declared - a physical ending or an
    already bound unresolved observation horizon.  Neither is reinterpreted.
    """
    events = attempt["events"] if events is None else events
    end_frame = attempt["owner_end_frame"] if end_frame is None else end_frame
    contacts = [float(row["frame"]) for row in events if row["event_type"] == "contact"]
    if not contacts:
        raise ValueError("connected scene requires at least one contact boundary")
    bounds = [*contacts, float(end_frame)]
    if bounds != sorted(bounds) or len(set(bounds)) != len(bounds):
        raise ValueError("ordered distinct original contact/endpoint domain required")
    return events, bounds


def inventory(
    attempt: dict,
    cameras_document: dict,
    *,
    events: list[dict] | None = None,
    end_frame: float | None = None,
    observation_partition: str = "fifth_frame_withheld",
    observation_fallback: bool = True,
) -> list[dict]:
    """Per-original-flight coverage of the supplied source rows.

    ``coverage_qualified`` is the unchanged four-training/one-check rule after
    the existing camera qualification.  Membership is the original membership:
    only the original terminal flight closes inclusively.
    """
    events, bounds = domain(attempt, events, end_frame)
    _, labels = visible_inputs(attempt, cameras_document, observation_fallback=observation_fallback)
    rows = []
    for index, (start, end) in enumerate(zip(bounds, bounds[1:])):
        inclusive_end = index == len(bounds) - 2
        native, train, check = frame_partition(
            labels,
            start,
            end,
            inclusive_end=inclusive_end,
            observation_partition=observation_partition,
        )
        rows.append(
            {
                "original_flight_index": index,
                "start_frame": float(start),
                "end_frame": float(end),
                "inclusive_end": bool(inclusive_end),
                "native_frames": [int(frame) for frame in native],
                "train_frames": [int(frame) for frame in train],
                "check_frames": [int(frame) for frame in check],
                "native_count": int(len(native)),
                "train_count": int(len(train)),
                "check_count": int(len(check)),
                "source_bounce_frames": [
                    float(row["frame"])
                    for row in events
                    if row["event_type"] == "bounce" and start < float(row["frame"]) <= end
                ],
                "coverage_qualified": bool(
                    len(train) >= TRAIN_MINIMUM and len(check) >= CHECK_MINIMUM
                ),
            }
        )
    return rows


def first_coverage_failure(rows: list[dict]) -> int | None:
    return next(
        (row["original_flight_index"] for row in rows if not row["coverage_qualified"]), None
    )
