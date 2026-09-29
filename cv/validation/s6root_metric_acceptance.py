"""Compare the unchanged pixel gate with an owner-facing metric plausibility gate.

This scorer is evaluation-only: owner bounce clicks are joined after reconstruction.
It does not alter the automatic Stage-6 acceptance decision.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from cv.pipeline import reconstruction
from cv.pipeline.flight_ledger import build as build_flight_ledger
from cv.validation.s6root_common import write_csv, write_json

NET_Y_M = 11.885
COURT_WIDTH_M = 10.97
NET_CENTER_HEIGHT_M = 0.914
NET_POST_HEIGHT_M = 1.07
BALL_RADIUS_M = 0.0325
BOUNCE_ERROR_LIMIT_M = 0.10
CONTACT_HEIGHT_RANGE_M = (0.30, 3.50)
CONTACT_REACH_LIMIT_M = 2.10
NET_CROSSING_MAX_HEIGHT_M = 4.50


def _data_root() -> Path:
    value = os.environ.get("TENNIS_DATA_ROOT")
    if not value:
        raise RuntimeError("TENNIS_DATA_ROOT is required")
    return Path(value).resolve()


def interpolate_net_crossing(trajectory: list[dict[str, Any]]) -> np.ndarray | None:
    """Interpolate the first fitted state crossing the physical net plane."""
    for before, after in zip(trajectory, trajectory[1:]):
        first = np.asarray(before.get("xyz"), dtype=float)
        second = np.asarray(after.get("xyz"), dtype=float)
        if first.shape != (3,) or second.shape != (3,):
            continue
        delta = float(second[1] - first[1])
        if abs(delta) < 1e-12 or (first[1] - NET_Y_M) * (second[1] - NET_Y_M) > 0.0:
            continue
        fraction = float((NET_Y_M - first[1]) / delta)
        if 0.0 <= fraction <= 1.0:
            return first + fraction * (second - first)
    return None


def net_height_m(x_m: float) -> float:
    lateral = min(1.0, abs(float(x_m) - COURT_WIDTH_M / 2.0) / (COURT_WIDTH_M / 2.0))
    return NET_CENTER_HEIGHT_M + lateral * (NET_POST_HEIGHT_M - NET_CENTER_HEIGHT_M)


def load_owner_bounces(
    truth_events: Path, camera_root: Path
) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    payload = json.loads(truth_events.read_text())
    output: dict[str, list[dict[str, Any]]] = {}
    truth_points = set()
    cameras: dict[tuple[str, str], reconstruction.PointCamera] = {}
    for row in payload["emissions"]:
        point = str(row["clip"])
        truth_points.add(point)
        if row.get("event_type") != "bounce":
            continue
        match_id, clip = point.rsplit("__", 1)
        key = (match_id, clip)
        if key not in cameras:
            try:
                cameras[key] = reconstruction.PointCamera(camera_root / match_id, clip)
            except ValueError:
                continue
        location = row["location"]
        pixel = np.asarray([location["image_x"], location["image_y"]], dtype=float)
        frame = float(row["frame"])
        court_xy = reconstruction.court_point(cameras[key], pixel, frame)
        output.setdefault(point, []).append(
            {"frame": frame, "court_xy": np.asarray(court_xy, dtype=float)}
        )
    return output, truth_points


def flight_metric_row(
    point: dict[str, Any],
    attempt: dict[str, Any],
    fit: dict[str, Any] | None,
    owner_bounces: list[dict[str, Any]],
    pixel_accepted: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "point": point["point"],
        "flight_index": int(attempt["flight_index"]),
        "solved": fit is not None,
        "pixel_gate_accepted": pixel_accepted,
    }
    if fit is None:
        return {
            **row,
            "metric_gate_accepted": False,
            "metric_reasons": "unsolved",
        }

    start = float(attempt["start_frame"])
    end = float(attempt["end_frame"])
    scoped_truth = [event for event in owner_bounces if start < float(event["frame"]) < end]
    fitted_bounces = fit.get("bounces", [])
    bounce_errors = []
    missing_bounce = False
    for truth in scoped_truth:
        candidates = [
            bounce
            for bounce in fitted_bounces
            if abs(float(bounce["frame"]) - float(truth["frame"])) <= 2.0
        ]
        if not candidates:
            missing_bounce = True
            continue
        selected = min(candidates, key=lambda bounce: abs(float(bounce["frame"]) - truth["frame"]))
        bounce_errors.append(
            float(
                np.linalg.norm(
                    np.asarray(selected["x"][:2], dtype=float)
                    - np.asarray(truth["court_xy"], dtype=float)
                )
            )
        )
    bounce_pass = not missing_bounce and all(
        error <= BOUNCE_ERROR_LIMIT_M for error in bounce_errors
    )

    net_constraint = fit.get("net_constraint")
    crossing = (
        np.asarray(net_constraint.get("xyz"), dtype=float)
        if isinstance(net_constraint, dict)
        and net_constraint.get("mode") == "plane_crossing_height_barrier"
        else interpolate_net_crossing(fit.get("trajectory", []))
    )
    if crossing is not None and crossing.shape != (3,):
        crossing = None
    crosses_sides = (
        attempt.get("start_side") in {"near", "far"}
        and attempt.get("end_side") in {"near", "far"}
        and attempt.get("start_side") != attempt.get("end_side")
    )
    crossing_required = bool(crosses_sides and not attempt.get("terminal_end"))
    crossing_height = float(crossing[2]) if crossing is not None else None
    net_pass = not crossing_required or (
        crossing is not None
        and (not isinstance(net_constraint, dict) or net_constraint.get("satisfied") is True)
        and crossing_height >= net_height_m(float(crossing[0])) + BALL_RADIUS_M - 0.01
        and crossing_height <= NET_CROSSING_MAX_HEIGHT_M
    )

    heights = [float(fit["start_xyz"][2])]
    reaches = [fit.get("start_player_distance_m")]
    if not attempt.get("terminal_end"):
        heights.append(float(fit["end_xyz"][2]))
        reaches.append(fit.get("end_player_distance_m"))
    contact_height_pass = all(
        CONTACT_HEIGHT_RANGE_M[0] <= value <= CONTACT_HEIGHT_RANGE_M[1] for value in heights
    )
    contact_reach_pass = all(
        value is not None and math.isfinite(float(value)) and float(value) <= CONTACT_REACH_LIMIT_M
        for value in reaches
    )
    reasons = []
    if scoped_truth and not bounce_pass:
        reasons.append("bounce_position")
    if not net_pass:
        reasons.append("net_crossing_height")
    if not contact_height_pass:
        reasons.append("contact_height")
    if not contact_reach_pass:
        reasons.append("contact_reach")
    return {
        **row,
        "owner_bounces": len(scoped_truth),
        "bounce_error_max_m": max(bounce_errors, default=None),
        "bounce_10cm_pass": None if not scoped_truth else bounce_pass,
        "net_crossing_required": crossing_required,
        "net_crossing_height_m": crossing_height,
        "net_crossing_plausible": net_pass,
        "contact_height_min_m": min(heights),
        "contact_height_max_m": max(heights),
        "contact_heights_m": heights,
        "contact_height_plausible": contact_height_pass,
        "contact_reach_max_m": (
            max(float(value) for value in reaches if value is not None)
            if any(value is not None for value in reaches)
            else None
        ),
        "contact_reaches_m": [float(value) for value in reaches if value is not None],
        "contact_reach_plausible": contact_reach_pass,
        "metric_gate_accepted": not reasons,
        "metric_reasons": ";".join(reasons),
    }


def run(
    report_path: Path, truth_events: Path, camera_root: Path, output_root: Path
) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    ledger = build_flight_ledger([report_path])
    pixel = {
        (row["point"], int(row["flight_index"])): row["status"] == "provisional_valid"
        for row in ledger["rows"]
    }
    owner_bounces, truth_points = load_owner_bounces(truth_events, camera_root)
    rows = []
    complete_pixel = set()
    complete_metric = set()
    for point in report["points_detail"]:
        point_id = str(point["point"])
        fits = {int(fit["flight_index"]): fit for fit in point.get("fits", [])}
        attempts = point.get("flight_attempts", [])
        point_rows = []
        for attempt in attempts:
            index = int(attempt["flight_index"])
            scored = flight_metric_row(
                point,
                attempt,
                fits.get(index),
                owner_bounces.get(point_id, []),
                pixel.get((point_id, index), False),
            )
            rows.append(scored)
            point_rows.append(scored)
        terminal = sum(bool(attempt.get("terminal_end")) for attempt in attempts)
        if attempts and terminal == 1 and all(row["pixel_gate_accepted"] for row in point_rows):
            complete_pixel.add(point_id)
        if attempts and terminal == 1 and all(row["metric_gate_accepted"] for row in point_rows):
            complete_metric.add(point_id)

    bounce_errors = [
        float(row["bounce_error_max_m"])
        for row in rows
        if row.get("bounce_error_max_m") is not None
    ]
    contact_heights = [float(value) for row in rows for value in row.get("contact_heights_m", [])]
    contact_reaches = [float(value) for row in rows for value in row.get("contact_reaches_m", [])]

    payload = {
        "schema": "s6root_metric_acceptance_v1",
        "artifact_class": "offline_truth_scoring",
        "automatic_gate_changed": False,
        "thresholds": {
            "bounce_error_m": BOUNCE_ERROR_LIMIT_M,
            "contact_height_m": CONTACT_HEIGHT_RANGE_M,
            "contact_reach_m": CONTACT_REACH_LIMIT_M,
            "net_crossing_max_height_m": NET_CROSSING_MAX_HEIGHT_M,
            "net_crossing_minimum": "local regulation net height plus ball radius minus 0.01 m",
        },
        "summary": {
            "flights": len(rows),
            "solved_flights": sum(row["solved"] for row in rows),
            "flights_with_owner_bounce": sum(bool(row.get("owner_bounces")) for row in rows),
            "bounce_10cm_pass_flights": sum(row.get("bounce_10cm_pass") is True for row in rows),
            "bounce_error_max_m_median_p90": (
                [float(np.median(bounce_errors)), float(np.percentile(bounce_errors, 90.0))]
                if bounce_errors
                else [None, None]
            ),
            "net_crossing_required_flights": sum(
                bool(row.get("net_crossing_required")) for row in rows
            ),
            "net_crossing_required_pass_flights": sum(
                bool(row.get("net_crossing_required")) and row.get("net_crossing_plausible") is True
                for row in rows
            ),
            "net_crossing_plausible_flights": sum(
                row.get("net_crossing_plausible") is True for row in rows
            ),
            "contact_height_plausible_flights": sum(
                row.get("contact_height_plausible") is True for row in rows
            ),
            "contact_height_m_p10_median_p90": (
                [
                    float(np.percentile(contact_heights, 10)),
                    float(np.median(contact_heights)),
                    float(np.percentile(contact_heights, 90)),
                ]
                if contact_heights
                else [None, None, None]
            ),
            "contact_reach_plausible_flights": sum(
                row.get("contact_reach_plausible") is True for row in rows
            ),
            "contact_reach_m_p10_median_p90": (
                [
                    float(np.percentile(contact_reaches, 10)),
                    float(np.median(contact_reaches)),
                    float(np.percentile(contact_reaches, 90)),
                ]
                if contact_reaches
                else [None, None, None]
            ),
            "pixel_gate_flights": sum(row["pixel_gate_accepted"] for row in rows),
            "metric_gate_flights": sum(row["metric_gate_accepted"] for row in rows),
            "both_gates_flights": sum(
                row["pixel_gate_accepted"] and row["metric_gate_accepted"] for row in rows
            ),
            "metric_only_flights": sum(
                row["metric_gate_accepted"] and not row["pixel_gate_accepted"] for row in rows
            ),
            "pixel_complete_points_over_198": len(complete_pixel),
            "metric_complete_points_over_198": len(complete_metric),
            "pixel_complete_points_over_146": len(complete_pixel & truth_points),
            "metric_complete_points_over_146": len(complete_metric & truth_points),
        },
        "rows": rows,
    }
    output_root.mkdir(parents=True, exist_ok=False)
    write_json(output_root / "metric_acceptance.json", payload)
    write_csv(
        output_root / "metric_acceptance_flights.csv",
        rows,
        sorted({key for row in rows for key in row}),
    )
    return payload


def parser() -> argparse.ArgumentParser:
    root = _data_root()
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--report", type=Path, required=True)
    result.add_argument(
        "--truth-events",
        type=Path,
        default=root / "processed/wk3_oracle/oracle_3d_ceiling_v2/truth_events_event_model_v3.json",
    )
    result.add_argument(
        "--camera-root",
        type=Path,
        default=root / "processed/wk3_s6fix/step3_arc_oracle_root",
    )
    result.add_argument("--output-root", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    payload = run(args.report, args.truth_events, args.camera_root, args.output_root)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
