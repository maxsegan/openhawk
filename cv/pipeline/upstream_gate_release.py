"""Apply the three upstream-gate releases while a packet is built.

The fitter never sees a row the emission gate abstained. These releases run on
the frozen emission document, before the inventory, and only when the policy
names them. With every switch off the document is not read again and not copied.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from cv.pipeline.abstained_release import POLICY_KEY as BAND_RELEASE_KEY
from cv.pipeline.abstained_release import release_supported_abstentions
from cv.pipeline.camera_frame_support import match_rows
from cv.pipeline.confident_hold_release import POLICY_KEY as HELD_RELEASE_KEY
from cv.pipeline.confident_hold_release import release_confident_holds
from cv.pipeline.event_impulse_support import SERVE_RELEASE_KEY, release_serve_contacts
from cv.pipeline.live_shot_camera import (
    POLICY_KEY,
    live_spans,
    release_camera_holds,
    rehold_closeups,
    shot_composition_hold,
)
from cv.pipeline.s6_contact_prefix_scope import PREFIX_ISOLATION_FIELD
from cv.pipeline.serve_toss_proposal import POLICY_KEY as SERVE_TOSS_KEY
from cv.pipeline.serve_toss_proposal import admit_serve_proposals

GATE_RELEASE_KEYS = (
    POLICY_KEY,
    SERVE_RELEASE_KEY,
    PREFIX_ISOLATION_FIELD,
    HELD_RELEASE_KEY,
    SERVE_TOSS_KEY,
    BAND_RELEASE_KEY,
)


def enabled(settings: dict | None, key: str) -> bool:
    return bool(settings) and settings.get(key) == "on"


def any_enabled(settings: dict | None) -> bool:
    return any(enabled(settings, key) for key in GATE_RELEASE_KEYS)


def _gate_row(path: Path, match_id: str, clip: str) -> dict | None:
    document = json.loads(path.read_text())
    for row in document["rows"]:
        if row.get("match_id") == match_id and row.get("clip") == clip:
            return row
    return None


def _arcs(path: Path, match_id: str, clip: str) -> list[dict]:
    document = json.loads(path.read_text())
    for row in document["rows"]:
        if row.get("match_id") == match_id and row.get("clip") == clip:
            return list(row.get("arcs") or [])
    return []


def _native_frame(value: str) -> int:
    return int(Path(value).stem.removeprefix("f_"))


def _track(path: Path, clip: str) -> dict[int, np.ndarray]:
    track: dict[int, np.ndarray] = {}
    with path.open() as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "x_native" not in reader.fieldnames:
            raise ValueError("serve release requires native ball columns")
        for row in reader:
            if row.get("clip") != clip:
                continue
            if "interpolated" in (row.get("sources") or ""):
                continue
            track[_native_frame(row["frame"])] = np.asarray(
                [float(row["x_native"]), float(row["y_native"])], dtype=float
            )
    return track


def _boxes(path: Path, clip: str) -> dict[int, list[tuple[str, tuple[float, float, float, float]]]]:
    boxes: dict[int, list] = {}
    with path.open() as handle:
        reader = csv.DictReader(handle)
        required = {"frame", "side", "x0_native", "y0_native", "x1_native", "y1_native", "clip"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError("serve release requires native player boxes")
        for row in reader:
            if row.get("clip") != clip:
                continue
            boxes.setdefault(_native_frame(row["frame"]), []).append(
                (
                    row["side"],
                    (
                        float(row["x0_native"]),
                        float(row["y0_native"]),
                        float(row["x1_native"]),
                        float(row["y1_native"]),
                    ),
                )
            )
    return boxes


def _cadence_row(upstream: Path, match_id: str, clip: str) -> dict | None:
    path = upstream / "frame_cadence_audit_v1.json"
    if not path.is_file():
        return None
    document = json.loads(path.read_text())
    for row in document.get("rows") or []:
        if row.get("match_id") == match_id and row.get("clip") == clip:
            return row
    return None


def load_context(
    *,
    match_id: str,
    clip: str,
    fps: float,
    events_path: Path,
    camera_path: Path,
    ball_path: Path,
    pose_path: Path,
) -> dict:
    """Read the frozen gate, camera support, track and boxes for one clip."""

    upstream = events_path.parent
    gate = _gate_row(upstream / "point_validity_gate_v1.json", match_id, clip)
    support = match_rows(camera_path.parent, match_id, [clip])[0]
    spans = live_spans(support["supported_spans"], fps)
    return {
        "match_id": match_id,
        "clip": clip,
        "fps": float(fps),
        "live_spans": spans,
        "release_camera": bool(spans) and shot_composition_hold(gate),
        "arcs": _arcs(upstream / "untouched_tracking_point_gate_v1.json", match_id, clip),
        "track": _track(ball_path, clip),
        "boxes": _boxes(pose_path, clip),
        "camera_spans": support["supported_spans"],
        "cadence": _cadence_row(upstream, match_id, clip),
    }


def before_operating_point(rows: list[dict], settings: dict | None, context: dict | None):
    """Clear in-span camera holds so the operating point can re-read what it may."""

    if not enabled(settings, POLICY_KEY) or context is None:
        return rows, None
    updated, census = release_camera_holds(
        rows,
        match_id=context["match_id"],
        clip=context["clip"],
        spans=context["live_spans"],
        release=context["release_camera"],
        fps=context["fps"],
    )
    return updated, census


def after_operating_point(rows: list[dict], settings: dict | None, context: dict | None):
    """Re-hold close-ups, release held serves and confident holds, then admit toss serves."""

    census: dict = {}
    if context is None:
        return rows, census
    if enabled(settings, POLICY_KEY):
        rows, held = rehold_closeups(
            rows,
            match_id=context["match_id"],
            clip=context["clip"],
            spans=context["live_spans"],
            fps=context["fps"],
        )
        census["closeup"] = held
    if enabled(settings, SERVE_RELEASE_KEY):
        rows, serves = release_serve_contacts(
            rows,
            match_id=context["match_id"],
            clip=context["clip"],
            arcs=context["arcs"],
            track=context["track"],
            boxes_by_frame=context["boxes"],
            fps=context["fps"],
        )
        census["serve"] = serves
    if enabled(settings, HELD_RELEASE_KEY):
        rows, confident = release_confident_holds(
            rows,
            match_id=context["match_id"],
            clip=context["clip"],
            camera_spans=context.get("camera_spans"),
            cadence=context.get("cadence"),
            release=True,
        )
        census["confident"] = confident
    # Last, so the releases above see only the decoder's own rows.
    if enabled(settings, SERVE_TOSS_KEY):
        rows, toss = admit_serve_proposals(
            rows,
            match_id=context["match_id"],
            clip=context["clip"],
            track=context["track"],
            boxes_by_frame=context["boxes"],
            fps=context["fps"],
        )
        census["serve_toss"] = toss
    return rows, census


def release_band_abstentions(rows: list[dict], settings: dict | None, context: dict | None, window):
    """Second tier: supported band abstentions inside the attempt window."""

    if not enabled(settings, BAND_RELEASE_KEY) or context is None:
        return rows, None
    return release_supported_abstentions(
        rows,
        match_id=context["match_id"],
        clip=context["clip"],
        window=window,
        context=context,
        release=True,
    )
