"""Build a label-blind quality ledger for automatic 3D ball flights."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

MIN_SPEED_KMH = 5.0
MAX_SPEED_KMH = 260.0
MAX_REPROJECTION_PX = 8.0
MAX_RECOVERABLE_REPROJECTION_PX = 20.0
MIN_OBSERVATION_COVERAGE = 0.20
MINIMUM_BALL_HEIGHT_M = 0.018
MAXIMUM_BOUNCE_CONTINUITY_M = 0.12
MAXIMUM_BOUNCE_SPEED_RATIO = 1.02
MAXIMUM_BOUNCE_IMPACT_VELOCITY_SLACK_MPS = 7.5
MAXIMUM_CONTACT_REACH_M = 2.1
MAXIMUM_CONTACT_REPROJECTION_PX = 12.0
MAXIMUM_CONTACT_APPARENT_HEIGHT_ERROR_M = 0.45
COURT_WIDTH_M = 10.97
NET_COURT_Y_M = 11.885
NET_CENTER_HEIGHT_M = 0.914
NET_POST_HEIGHT_M = 1.07
BALL_RADIUS_M = 0.0325
MINIMUM_NET_CLEARANCE_M = -0.01


def finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fallback_attempts(point: dict) -> list[dict]:
    fits = point.get("fits", [])
    attempts = [
        {
            "flight_index": int(fit.get("flight_index", index)),
            "start_frame": fit.get("start_frame"),
            "end_frame": fit.get("end_frame"),
            "duration_seconds": (
                (float(fit["end_frame"]) - float(fit["start_frame"])) / float(point["fps"])
                if fit.get("start_frame") is not None and fit.get("end_frame") is not None
                else None
            ),
            "start_side": None,
            "end_side": None,
            "start_phase": None,
            "terminal_end": bool(fit.get("terminal_end")),
            "intermediate_events": [],
            "solved": True,
            "fit_index": index,
            "legacy_attempt_reconstruction": True,
        }
        for index, fit in enumerate(fits)
    ]
    missing = max(0, int(point.get("attempted_shots", len(attempts))) - len(attempts))
    attempts.extend(
        {
            "flight_index": len(attempts) + index,
            "start_frame": None,
            "end_frame": None,
            "duration_seconds": None,
            "start_side": None,
            "end_side": None,
            "start_phase": None,
            "terminal_end": False,
            "intermediate_events": [],
            "solved": False,
            "legacy_attempt_reconstruction": True,
        }
        for index in range(missing)
    )
    return attempts


def net_clearance_m(fit: dict) -> float | None:
    """Interpolate the fitted ball-center clearance at the physical net plane."""
    constraint = fit.get("net_constraint")
    if (
        isinstance(constraint, dict)
        and constraint.get("mode") == "plane_crossing_height_barrier"
        and finite_float(constraint.get("clearance_m")) is not None
    ):
        return finite_float(constraint["clearance_m"])
    trajectory = fit.get("trajectory", [])
    clearances = []
    for before, after in zip(trajectory, trajectory[1:]):
        first = before.get("xyz")
        second = after.get("xyz")
        if not first or not second:
            continue
        y0, y1 = float(first[1]), float(second[1])
        if y0 == y1 or (y0 - NET_COURT_Y_M) * (y1 - NET_COURT_Y_M) > 0.0:
            continue
        fraction = (NET_COURT_Y_M - y0) / (y1 - y0)
        if not 0.0 <= fraction <= 1.0:
            continue
        x = float(first[0]) + fraction * (float(second[0]) - float(first[0]))
        z = float(first[2]) + fraction * (float(second[2]) - float(first[2]))
        lateral = min(1.0, abs(x - COURT_WIDTH_M / 2.0) / (COURT_WIDTH_M / 2.0))
        net_height = NET_CENTER_HEIGHT_M + lateral * (NET_POST_HEIGHT_M - NET_CENTER_HEIGHT_M)
        clearances.append(z - (net_height + BALL_RADIUS_M))
    return min(clearances) if clearances else None


def classify_flight(point: dict, attempt: dict, fit: dict | None) -> tuple[str, list[str]]:
    if fit is None:
        return "unsolved", ["physics_fit_failed"]

    reasons = []
    rms = finite_float(fit.get("rms_px"))
    anchor_first = str(fit.get("fit_method", "")).startswith("anchor_first_")
    held_out_median = finite_float(fit.get("held_out_reprojection_median_px"))
    held_out_p90 = finite_float(fit.get("held_out_reprojection_p90_px"))
    speed = finite_float(fit.get("speed_kmh"))
    observation_coverage = finite_float(fit.get("observation_coverage"))
    if observation_coverage is None:
        duration_frames = (
            float(attempt["end_frame"]) - float(attempt["start_frame"]) + 1.0
            if attempt.get("start_frame") is not None and attempt.get("end_frame") is not None
            else None
        )
        if duration_frames:
            observation_coverage = float(fit.get("observations", 0)) / duration_frames

    if anchor_first:
        if held_out_median is None or held_out_median > MAX_REPROJECTION_PX:
            reasons.append("high_held_out_reprojection")
        if held_out_p90 is None or held_out_p90 > MAXIMUM_CONTACT_REPROJECTION_PX:
            reasons.append("high_held_out_reprojection_p90")
        if fit.get("anchor_satisfied") is not True:
            reasons.append("anchor_violation")
    elif rms is None or rms > MAX_REPROJECTION_PX:
        reasons.append("high_reprojection")
    if speed is None or not MIN_SPEED_KMH <= speed <= MAX_SPEED_KMH:
        reasons.append("implausible_speed")
    if observation_coverage is None or observation_coverage < MIN_OBSERVATION_COVERAGE:
        reasons.append("sparse_observations")
    bounce_count = len(fit.get("bounces", []))
    if bounce_count > 2 or (bounce_count > 1 and not attempt.get("terminal_end")):
        reasons.append("multiple_bounces")
    minimum_height = finite_float(fit.get("minimum_height_m"))
    if minimum_height is not None and minimum_height < MINIMUM_BALL_HEIGHT_M:
        reasons.append("physics_underground")
    bounce_continuity = finite_float(fit.get("fixed_bounce_continuity_m"))
    if bounce_continuity is not None and bounce_continuity > MAXIMUM_BOUNCE_CONTINUITY_M:
        reasons.append("physics_bounce_discontinuity")
    bounce_energy_ratio = finite_float(fit.get("bounce_energy_ratio"))
    if bounce_energy_ratio is None:
        bounce_energy_ratio = finite_float(fit.get("bounce_speed_ratio"))
    if bounce_energy_ratio is not None and bounce_energy_ratio > MAXIMUM_BOUNCE_SPEED_RATIO:
        reasons.append("physics_bounce_energy_gain")
    bounce_impact_slack = finite_float(fit.get("bounce_impact_velocity_slack_mps"))
    if bounce_impact_slack is not None and (
        bounce_impact_slack > MAXIMUM_BOUNCE_IMPACT_VELOCITY_SLACK_MPS
    ):
        reasons.append("physics_bounce_impact_velocity_slack")
    player_distances = [
        value
        for key in ("start_player_distance_m", "end_player_distance_m")
        if (value := finite_float(fit.get(key))) is not None
    ]
    if player_distances and max(player_distances) > MAXIMUM_CONTACT_REACH_M:
        reasons.append("physics_contact_reach")
    contact_reprojections = [
        value
        for key in ("start_contact_reprojection_px", "end_contact_reprojection_px")
        if (value := finite_float(fit.get(key))) is not None
    ]
    if contact_reprojections and max(contact_reprojections) > MAXIMUM_CONTACT_REPROJECTION_PX:
        reasons.append("physics_contact_reprojection")
    if any(
        (fit.get(f"{endpoint}_contact_reprojection") or {}).get("position_available") is False
        for endpoint in ("start", "end")
    ):
        reasons.append("physics_contact_observation_uncovered")
    contact_height_errors = [
        value
        for key in (
            "start_contact_apparent_height_error_m",
            "end_contact_apparent_height_error_m",
        )
        if (value := finite_float(fit.get(key))) is not None
    ]
    if contact_height_errors and (
        max(contact_height_errors) > MAXIMUM_CONTACT_APPARENT_HEIGHT_ERROR_M
    ):
        reasons.append("physics_contact_body_scale")
    clearance = finite_float(fit.get("net_clearance_m"))
    if clearance is None:
        clearance = net_clearance_m(fit)
    explicit_net = any(
        event.get("event_type") == "net_hit" for event in attempt.get("intermediate_events", [])
    )
    if clearance is not None and clearance < MINIMUM_NET_CLEARANCE_M and not explicit_net:
        reasons.append("physics_net_penetration_without_collision")
    if (
        not attempt.get("terminal_end")
        and attempt.get("start_side") in {"near", "far"}
        and attempt.get("start_side") == attempt.get("end_side")
    ):
        reasons.append("same_side_contacts")
    if not point.get("active_play_valid", True):
        reasons.append("active_play_invalid")
    smoothing = point.get("smoothing", {})
    if (finite_float(smoothing.get("post_heal_teleport_rate", 0.0)) or 0.0) > 0.02:
        reasons.append("tracking_teleport")
    branches = point.get("interpretation_branches", {})
    if branches.get("enabled") and branches.get("decisive") is not True:
        reasons.append("branch_ambiguous")

    if not reasons:
        return "provisional_valid", []
    if (
        (held_out_median if anchor_first else rms) is not None
        and (held_out_median if anchor_first else rms) <= MAX_RECOVERABLE_REPROJECTION_PX
        and speed is not None
        and MIN_SPEED_KMH <= speed <= MAX_SPEED_KMH
    ):
        return "recoverable", reasons
    return "invalid", reasons


def diagnostic_confidence(status: str, fit: dict | None) -> float:
    if fit is None:
        return 0.0
    rms = finite_float(
        fit.get("held_out_reprojection_median_px")
        if str(fit.get("fit_method", "")).startswith("anchor_first_")
        else fit.get("rms_px")
    )
    rms = 100.0 if rms is None else rms
    coverage = finite_float(fit.get("observation_coverage"))
    if coverage is None:
        coverage = min(1.0, float(fit.get("observations", 0)) / 20.0)
    score = math.exp(-rms / 8.0) * min(1.0, max(0.0, coverage))
    if status != "provisional_valid":
        score *= 0.5
    return float(min(0.999, max(0.0, score)))


def point_rows(point: dict, cohort: str) -> list[dict]:
    fits = point.get("fits", [])
    fits_by_index = {int(fit.get("flight_index", index)): fit for index, fit in enumerate(fits)}
    attempts = point.get("flight_attempts") or fallback_attempts(point)
    rows = []
    for attempt in attempts:
        flight_index = int(attempt["flight_index"])
        fit = fits_by_index.get(flight_index)
        if skip_reason := attempt.get("skip_reason"):
            status, reasons = "unsolved", [str(skip_reason)]
        else:
            status, reasons = classify_flight(point, attempt, fit)
        rows.append(
            {
                "flight_id": f"{point['point']}__flight_{flight_index:03d}",
                "cohort": cohort,
                "point": point["point"],
                "match_id": point["match_id"],
                "clip": point["clip"],
                "flight_index": flight_index,
                "fps": float(point["fps"]),
                "surface": point.get("surface"),
                "start_frame": attempt.get("start_frame"),
                "end_frame": attempt.get("end_frame"),
                "duration_seconds": attempt.get("duration_seconds"),
                "start_side": attempt.get("start_side"),
                "end_side": attempt.get("end_side"),
                "start_phase": attempt.get("start_phase"),
                "terminal_end": bool(attempt.get("terminal_end")),
                "point_decision": point.get("decision"),
                "point_reasons": point.get("reasons", []),
                "solved": fit is not None,
                "status": status,
                "reasons": reasons,
                "diagnostic_confidence": diagnostic_confidence(status, fit),
                "rms_px": fit.get("rms_px") if fit else None,
                "weighted_rms_px": fit.get("weighted_rms_px") if fit else None,
                "fit_method": fit.get("fit_method") if fit else None,
                "held_out_reprojection_median_px": (
                    fit.get("held_out_reprojection_median_px") if fit else None
                ),
                "held_out_reprojection_p90_px": (
                    fit.get("held_out_reprojection_p90_px") if fit else None
                ),
                "anchor_max_error_m": fit.get("anchor_max_error_m") if fit else None,
                "anchor_satisfied": fit.get("anchor_satisfied") if fit else None,
                "spin_identifiable": fit.get("spin_identifiable") if fit else None,
                "observations": fit.get("observations") if fit else None,
                "observation_coverage": (fit.get("observation_coverage") if fit else None),
                "speed_kmh": fit.get("speed_kmh") if fit else None,
                "spin_rpm": fit.get("spin_rpm") if fit else None,
                "bounce_count": len(fit.get("bounces", [])) if fit else None,
                "minimum_height_m": fit.get("minimum_height_m") if fit else None,
                "fixed_bounce_continuity_m": (
                    fit.get("fixed_bounce_continuity_m") if fit else None
                ),
                "bounce_speed_ratio": fit.get("bounce_speed_ratio") if fit else None,
                "start_player_distance_m": (fit.get("start_player_distance_m") if fit else None),
                "end_player_distance_m": fit.get("end_player_distance_m") if fit else None,
                "start_contact_reprojection_px": (
                    fit.get("start_contact_reprojection_px") if fit else None
                ),
                "end_contact_reprojection_px": (
                    fit.get("end_contact_reprojection_px") if fit else None
                ),
                "start_contact_apparent_height_error_m": (
                    fit.get("start_contact_apparent_height_error_m") if fit else None
                ),
                "end_contact_apparent_height_error_m": (
                    fit.get("end_contact_apparent_height_error_m") if fit else None
                ),
                "net_clearance_m": net_clearance_m(fit) if fit else None,
                "intermediate_events": attempt.get("intermediate_events", []),
                "start_boundary": attempt.get("start_boundary"),
                "end_boundary": attempt.get("end_boundary"),
                "legacy_attempt_reconstruction": bool(attempt.get("legacy_attempt_reconstruction")),
            }
        )
    return rows


def build(report_paths: list[Path]) -> dict:
    rows = []
    reports = []
    points = []
    flight_groups = []
    for path in report_paths:
        report = json.loads(path.read_text())
        reports.append(str(path))
        cohort = path.parents[1].name if len(path.parents) > 1 else path.stem
        for point in report.get("points_detail", []):
            points.append(point)
            group = point_rows(point, cohort)
            flight_groups.append(group)
            rows.extend(group)
    counts = Counter(row["status"] for row in rows)
    solved = sum(row["solved"] for row in rows)
    complete_point_candidates = sum(
        bool(point_rows)
        and point_rows[0].get("point_decision") == "retain"
        and all(row["status"] == "provisional_valid" for row in point_rows)
        and sum(bool(row["terminal_end"]) for row in point_rows) == 1
        for point_rows in flight_groups
    )
    return {
        "schema": "automatic_3d_flight_ledger_v2",
        "point_denominator": (
            "all source reconstruction attempt records, including zero-flight attempts; "
            "source reports remain separate measurements, not independently counted scoring points"
        ),
        "owner_truth_loaded": False,
        "certification": (
            "provisional_valid is a label-blind physics/grammar screen; anchor-first rows use "
            "held-out checkerboard reprojection, but offline owner 3D scoring remains separate"
        ),
        "thresholds": {
            "minimum_speed_kmh": MIN_SPEED_KMH,
            "maximum_speed_kmh": MAX_SPEED_KMH,
            "maximum_reprojection_px": MAX_REPROJECTION_PX,
            "maximum_held_out_p90_px": MAXIMUM_CONTACT_REPROJECTION_PX,
            "minimum_observation_coverage": MIN_OBSERVATION_COVERAGE,
            "minimum_ball_height_m": MINIMUM_BALL_HEIGHT_M,
            "maximum_bounce_continuity_m": MAXIMUM_BOUNCE_CONTINUITY_M,
            "maximum_bounce_energy_ratio": MAXIMUM_BOUNCE_SPEED_RATIO,
            "maximum_bounce_impact_velocity_slack_mps": (MAXIMUM_BOUNCE_IMPACT_VELOCITY_SLACK_MPS),
            "maximum_contact_reach_m": MAXIMUM_CONTACT_REACH_M,
            "maximum_contact_reprojection_px": MAXIMUM_CONTACT_REPROJECTION_PX,
            "maximum_contact_apparent_height_error_m": (MAXIMUM_CONTACT_APPARENT_HEIGHT_ERROR_M),
        },
        "sources": reports,
        "summary": {
            "flights": len(rows),
            "solved": solved,
            "solve_rate": solved / len(rows) if rows else None,
            "provisional_valid": counts["provisional_valid"],
            "provisional_valid_rate": (counts["provisional_valid"] / len(rows) if rows else None),
            "recoverable": counts["recoverable"],
            "invalid": counts["invalid"],
            "unsolved": counts["unsolved"],
            "points": len(points),
            "flight_bearing_points": sum(bool(group) for group in flight_groups),
            "points_without_flights": sum(not group for group in flight_groups),
            "complete_point_candidates": complete_point_candidates,
            "matches": len({point["match_id"] for point in points}),
        },
        "rows": rows,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "flight_id",
        "cohort",
        "point",
        "match_id",
        "clip",
        "flight_index",
        "fps",
        "surface",
        "start_frame",
        "end_frame",
        "duration_seconds",
        "start_side",
        "end_side",
        "start_phase",
        "terminal_end",
        "point_decision",
        "point_reasons",
        "solved",
        "status",
        "reasons",
        "diagnostic_confidence",
        "rms_px",
        "weighted_rms_px",
        "fit_method",
        "held_out_reprojection_median_px",
        "held_out_reprojection_p90_px",
        "anchor_max_error_m",
        "anchor_satisfied",
        "spin_identifiable",
        "observations",
        "observation_coverage",
        "speed_kmh",
        "spin_rpm",
        "bounce_count",
        "minimum_height_m",
        "fixed_bounce_continuity_m",
        "bounce_speed_ratio",
        "start_player_distance_m",
        "end_player_distance_m",
        "start_contact_reprojection_px",
        "end_contact_reprojection_px",
        "start_contact_apparent_height_error_m",
        "end_contact_apparent_height_error_m",
        "net_clearance_m",
        "legacy_attempt_reconstruction",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row.get(key) for key in fieldnames},
                    "reasons": "|".join(row["reasons"]),
                    "point_reasons": "|".join(row["point_reasons"]),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    ledger = build(args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    if args.csv:
        write_csv(args.csv, ledger["rows"])
    print(json.dumps(ledger["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
