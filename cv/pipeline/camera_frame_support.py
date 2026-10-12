"""Publish the per-frame reliable-camera support a retained-play event scope may use.

An active-play span is a shot-composition verdict.  It names the frames the producer
believes are live play; it does not certify that the broadcast is on the standard
elevated court view there.  A supplied clip can carry a close-up inside a shot whose
``is_play_camera`` flag is true, so a span alone is not per-frame view evidence.

This producer adds no threshold of its own.  It reads the existing
``reliable_per_frame`` court transport -- the same automatic
``court_H_per_frame_v1.npz`` acceptance the event feature stage already uses, where a
frame is accepted only when the registration is reliable and its native-image ground
homography is finite and invertible -- and republishes, per point, the closed native
frame runs whose camera view is supported.  A clip with no artifact, or with no
accepted frame, publishes an empty support set and an explicit reason, so a consumer
that requires support fails closed rather than assuming it.

    python -m cv.pipeline.camera_frame_support --manifest <manifest.json> \
        --out <processed root> \
        --output <processed root>/reliable_per_frame_camera_support_v1.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from cv.pipeline import provenance
from cv.pipeline.event_model_v2_features import (
    FRAME_HOMOGRAPHY_NAME,
    load_court_transport,
)
from cv.pipeline.point_validity import CAMERA_SUPPORT_TRANSPORT

SCHEMA = "reliable_per_frame_camera_support_v1"
# One declaration, shared with the consumer that publishes it in its scope record.
# The retained-play scope wants measured per-frame support, so a missing frame is a
# missing frame: the point-static fallback is deliberately not consulted here.
TRANSPORT_MODE = CAMERA_SUPPORT_TRANSPORT["mode"]
TRANSPORT_MISSING = CAMERA_SUPPORT_TRANSPORT["missing"]
MISSING_ARTIFACT_REASON = "missing_frame_transport_artifact"
NO_SUPPORTED_FRAME_REASON = "no_reliable_per_frame_camera_frame"


def closed_runs(frames: Iterable[int]) -> list[list[int]]:
    """Collapse native frame integers into ordered, disjoint closed ``[low, high]`` runs."""
    runs: list[list[int]] = []
    for frame in sorted({int(value) for value in frames}):
        if runs and frame == runs[-1][1] + 1:
            runs[-1][1] = frame
        else:
            runs.append([frame, frame])
    return runs


def match_rows(match_root: Path, match_id: str, clips: Iterable[str]) -> list[dict]:
    """Return one support row per requested clip of one match root."""
    transport = load_court_transport(match_root, mode=TRANSPORT_MODE, missing=TRANSPORT_MISSING)
    artifact = transport.artifact
    rows = []
    for clip in clips:
        spans = closed_runs(transport.frames.get(clip, {}))
        reason = (
            MISSING_ARTIFACT_REASON
            if artifact is None
            else NO_SUPPORTED_FRAME_REASON
            if not spans
            else None
        )
        rows.append(
            {
                "point": f"{match_id}/{clip}",
                "match_id": match_id,
                "clip": clip,
                "artifact_present": artifact is not None,
                "supported_frames": sum(high - low + 1 for low, high in spans),
                "supported_spans": spans,
                **({"reason": reason} if reason is not None else {}),
            }
        )
    return rows


def build_report(manifest: dict, output_root: Path) -> dict:
    rows: list[dict] = []
    artifacts: list[dict] = []
    for match in manifest["matches"]:
        match_id = match["id"]
        match_root = output_root / match_id
        point_ids = match.get("point_ids", range(1, int(manifest["points_per_match"]) + 1))
        rows.extend(match_rows(match_root, match_id, [f"pt{point:04d}" for point in point_ids]))
        path = match_root / FRAME_HOMOGRAPHY_NAME
        if path.is_file():
            artifacts.append(provenance.file_record(path))
    return {
        "schema": SCHEMA,
        "labels_loaded": False,
        "transport": {**CAMERA_SUPPORT_TRANSPORT, "artifact_name": FRAME_HOMOGRAPHY_NAME},
        "frame_index_origin": 1,
        "artifacts": artifacts,
        "points": len(rows),
        "supported_points": sum(bool(row["supported_spans"]) for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(json.loads(args.manifest.read_text()), args.out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"reliable per-frame camera support {report['supported_points']}/{report['points']} "
        f"points -> {args.output}"
    )
    for row in report["rows"]:
        if not row["supported_spans"]:
            print(f"  {row['point']}: {row.get('reason')}")


if __name__ == "__main__":
    main()
