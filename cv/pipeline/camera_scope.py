"""Convert automatic camera-leakage evidence into split spatial/timing scope."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cv.pipeline.provenance import AUTOMATIC_MODE, build_provenance, file_record

SCHEMA = "camera_scope_v1"
SCOPE_VALUES = {"standard", "nonstandard_spatial", "camera_cut", "unknown"}


def _confidence(reasons: list[str]) -> float:
    strong = {"court_support_lost", "giant_player_box"}
    return 0.95 if strong.issubset(reasons) else 0.75


def _clip_interval(start: float, end: float, low: float, high: float) -> list[float] | None:
    clipped = [max(float(start), float(low)), min(float(end), float(high))]
    return clipped if clipped[0] <= clipped[1] else None


def _subtract_interval(span: list[float], exclusions: list[list[float]]) -> list[list[float]]:
    remaining = [[float(span[0]), float(span[1])]]
    for exclusion in exclusions:
        updated = []
        for start, end in remaining:
            overlap = _clip_interval(exclusion[0], exclusion[1], start, end)
            if overlap is None:
                updated.append([start, end])
                continue
            if start <= overlap[0] - 1.0:
                updated.append([start, overlap[0] - 1.0])
            if overlap[1] + 1.0 <= end:
                updated.append([overlap[1] + 1.0, end])
        remaining = updated
    return [interval for interval in remaining if interval[1] > interval[0]]


def point_scope(entry: dict, proposal: dict) -> dict:
    active_spans = entry.get("active_spans", [])
    event_spans = entry.get("event_spans", active_spans)
    exclusions = []
    for trim in proposal.get("proposed_trims", []):
        for active_start, active_end in active_spans:
            overlap = _clip_interval(
                trim["start_frame"],
                trim["end_frame"],
                active_start,
                active_end,
            )
            if overlap is not None:
                exclusions.append({"interval": overlap, "trim": trim})

    evidence_healthy = bool(proposal.get("court_arm_enabled") or proposal.get("box_arm_enabled"))
    intervals = []
    exclusion_intervals = [row["interval"] for row in exclusions]
    for active_span in active_spans:
        for interval in _subtract_interval(active_span, exclusion_intervals):
            intervals.append(
                {
                    "frame_scope": interval,
                    "camera_scope": "standard" if evidence_healthy else "unknown",
                    "spatial_usable": True if evidence_healthy else None,
                    "timing_usable": True,
                    "confidence": 0.70 if evidence_healthy else 0.0,
                    "reasons": [
                        "automatic_standard_camera_evidence"
                        if evidence_healthy
                        else "camera_scope_evidence_unhealthy"
                    ],
                }
            )
    for exclusion in exclusions:
        reasons = sorted(set(exclusion["trim"].get("reasons", [])))
        intervals.append(
            {
                "frame_scope": exclusion["interval"],
                "camera_scope": "nonstandard_spatial",
                "spatial_usable": False,
                "timing_usable": True,
                "confidence": _confidence(reasons),
                "reasons": reasons,
            }
        )
    intervals.sort(key=lambda row: (row["frame_scope"][0], row["frame_scope"][1]))
    calibration = proposal.get("calibration_provenance", {})
    return {
        "clip": entry.get("clip", proposal.get("clip")),
        "fps": float(entry.get("fps", proposal.get("fps", 0.0))),
        "active_spans": active_spans,
        "timing_context_spans": event_spans,
        "intervals": intervals,
        "spatial_withheld_frames": sum(
            max(0, int(end) - int(start) + 1)
            for start, end in (
                row["frame_scope"] for row in intervals if row["spatial_usable"] is False
            )
        ),
        "evidence_arms": {
            "court": "healthy" if proposal.get("court_arm_enabled") else "abstain",
            "player_box": "healthy" if proposal.get("box_arm_enabled") else "abstain",
        },
        "calibration_provenance": {
            "source": calibration.get("source", "court_H_per_point"),
            "residuals": calibration.get("residuals"),
            "frame_scope": calibration.get("frame_scope", active_spans),
            "fallback_ancestry": calibration.get("fallback_ancestry", []),
        },
    }


def window_scope(point: dict | None, start: float, end: float) -> dict:
    if point is None:
        return {
            "decision": "unknown",
            "spatial_usable": None,
            "timing_usable": True,
            "reasons": ["camera_scope_artifact_missing"],
        }
    overlapping = [
        row
        for row in point.get("intervals", [])
        if not (float(end) < row["frame_scope"][0] or float(start) > row["frame_scope"][1])
    ]
    if any(row["spatial_usable"] is False for row in overlapping):
        return {
            "decision": "withhold_spatial",
            "spatial_usable": False,
            "timing_usable": all(row["timing_usable"] for row in overlapping),
            "reasons": sorted(
                {
                    reason
                    for row in overlapping
                    if row["spatial_usable"] is False
                    for reason in row["reasons"]
                }
            ),
        }
    if overlapping and all(row["spatial_usable"] is True for row in overlapping):
        return {
            "decision": "retain_spatial",
            "spatial_usable": True,
            "timing_usable": True,
            "reasons": [],
        }
    return {
        "decision": "unknown",
        "spatial_usable": None,
        "timing_usable": True,
        "reasons": ["camera_scope_unknown"],
    }


def build_contract(active: dict, leakage_reports: list[dict]) -> dict:
    proposals = {}
    source_schemas = set()
    for report in leakage_reports:
        source_schemas.add(report.get("schema", "unknown"))
        match_id = report["match_id"]
        for clip, proposal in report.get("points", {}).items():
            proposals[f"{match_id}/{clip}"] = proposal
    points = {}
    for key, entry in sorted(active.items()):
        points[key] = point_scope(entry, proposals.get(key, {"clip": entry.get("clip")}))
    return {
        "schema": SCHEMA,
        "labels_loaded": False,
        "source_schemas": sorted(source_schemas),
        "points": points,
        "summary": {
            "points": len(points),
            "points_with_spatial_withholding": sum(
                row["spatial_withheld_frames"] > 0 for row in points.values()
            ),
            "spatial_withheld_frames": sum(
                row["spatial_withheld_frames"] for row in points.values()
            ),
            "points_unknown": sum(
                any(interval["spatial_usable"] is None for interval in row["intervals"])
                for row in points.values()
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-play", type=Path, required=True)
    parser.add_argument("--leakage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report_paths = sorted(args.leakage_root.glob("*__play_camera_leakage_v1.json"))
    reports = [json.loads(path.read_text()) for path in report_paths]
    contract = build_contract(json.loads(args.active_play.read_text()), reports)
    contract["provenance"] = build_provenance(
        root=Path(__file__).resolve().parents[2],
        mode=AUTOMATIC_MODE,
        configuration={
            "stage": "camera_scope",
            "schema": SCHEMA,
            "labels_loaded": False,
        },
        reused_artifacts=[
            file_record(args.active_play, role="active_play"),
            *[file_record(path, role="camera_leakage_evidence") for path in report_paths],
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(json.dumps(contract["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
