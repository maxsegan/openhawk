"""Compose automatic evidence artifacts into the production point gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cv.pipeline.point_validity import compose_point_gate

CAMERA_SUPPORT_SCHEMA = "reliable_per_frame_camera_support_v1"


def camera_support_rows(document: dict) -> list[dict]:
    """Read the declared per-frame camera support document, refusing any other schema."""
    if document.get("schema") != CAMERA_SUPPORT_SCHEMA:
        raise ValueError(f"unsupported camera support schema: {document.get('schema')!r}")
    return document["rows"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--active-play",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--tracking-gate",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument("--court-geometry", type=Path)
    parser.add_argument("--frame-cadence", type=Path)
    def invalid_fraction(value: str):
        # "absolute" disables the cohort-relative budget: points are held only for
        # their own hard reasons or the tracking gate's per-point decision.
        return None if value == "absolute" else float(value)

    parser.add_argument("--maximum-invalid-fraction", type=invalid_fraction, default=0.2)
    parser.add_argument(
        "--retained-play-scope",
        action="store_true",
        help=(
            "retain a point held only for phase:multiple_camera_shots when the court "
            "geometry is explicitly valid, the active-play record certifies a "
            "non-empty retained live-play interval, and the reliable per-frame camera "
            "support overlaps it; publish both so the event gate can scope acceptance "
            "to them; default off reproduces the previous document exactly"
        ),
    )
    parser.add_argument(
        "--camera-frame-support",
        type=Path,
        help=(
            "reliable_per_frame_camera_support_v1 document from "
            "cv.pipeline.camera_frame_support; required by --retained-play-scope, "
            "which never falls back to a span-only verdict"
        ),
    )
    args = parser.parse_args()
    if args.retained_play_scope and args.camera_frame_support is None:
        parser.error("--retained-play-scope requires --camera-frame-support")
    if args.camera_frame_support is not None and not args.retained_play_scope:
        parser.error("--camera-frame-support is only read by --retained-play-scope")
    active_play = json.loads(args.active_play.read_text())
    tracking_gate = json.loads(args.tracking_gate.read_text())
    court_geometry = (
        json.loads(args.court_geometry.read_text())["rows"]
        if args.court_geometry
        else None
    )
    frame_cadence = (
        json.loads(args.frame_cadence.read_text())["rows"]
        if args.frame_cadence
        else None
    )
    camera_support = (
        camera_support_rows(json.loads(args.camera_frame_support.read_text()))
        if args.camera_frame_support
        else None
    )
    report = compose_point_gate(
        active_play,
        tracking_gate["rows"],
        maximum_invalid_fraction=args.maximum_invalid_fraction,
        court_geometry_rows=court_geometry,
        frame_cadence_rows=frame_cadence,
        retained_play_scope_enabled=args.retained_play_scope,
        camera_support_rows=camera_support,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"held {report['held']}/{report['points']} points "
        f"({report['held'] / report['points']:.1%}) -> {args.output}"
    )
    if args.retained_play_scope:
        print(
            f"  retained play scope on: {report['play_interval_scoped_points']} "
            f"point(s) scoped to their retained play interval"
        )
        support = report["play_scope_camera_support"]
        print(
            f"    reliable per-frame camera support covers "
            f"{support['supported_play_span_frames']}/{support['play_span_frames']} "
            f"scoped live-play frames"
        )
        for row in report["rows"]:
            if row.get("play_scope") is not None:
                print(
                    f"    {row['point']}: spans={row['play_scope_native_spans']} "
                    f"camera_supported={row['play_scope_camera_supported_spans']}"
                )
    for row in report["rows"]:
        if row["decision"] == "hold":
            print(
                f"  {row['point']}: {','.join(row['reasons'])} "
                f"risk={row['tracking_risk']:.3f}"
            )


if __name__ == "__main__":
    main()
