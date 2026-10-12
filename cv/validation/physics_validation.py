"""Validate literature-reference physics against the anchor-first Stage-6 fitter.

This module is evaluation-only.  It locally substitutes the reference sampler and
bounce law into already-imported fitter symbols; it never changes automatic pipeline
code.  Outputs are written below ``$TENNIS_DATA_ROOT/processed/wk3_physics``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy.optimize import brentq

from cv.pipeline import anchor_first_fit, rich_ball_physics
from cv.validation import s6root_closed_loop
from cv.validation.s6root_common import (
    ORACLE_RELATIVE,
    FlightCase,
    compact_and_classify,
    load_flight_cases,
    percentile,
    write_csv,
    write_json,
)
from physics import flight as current_flight
from physics import impact as current_impact
from physics import reference

SEED = 20260902
TARGET_CLOSED_LOOP_FLIGHTS = 20
OUTPUT_RELATIVE = Path("processed/wk3_physics")


def data_root() -> Path:
    value = os.environ.get("TENNIS_DATA_ROOT")
    if not value:
        raise RuntimeError("TENNIS_DATA_ROOT is required")
    return Path(value).resolve()


def _reference_bounce_velocity(
    velocity: np.ndarray, spin: np.ndarray, surface: str
) -> tuple[np.ndarray, np.ndarray, str]:
    result = reference.court_bounce(velocity, spin, surface)
    return result.velocity, result.spin, result.regime


@contextmanager
def reference_fitter_adapter() -> Iterator[None]:
    """Temporarily route fitter-owned physics calls through ``physics.reference``."""

    replacements = (
        (anchor_first_fit, "sample_states", reference.sample_states),
        (anchor_first_fit, "sample_states_batch", reference.sample_states_batch),
        (anchor_first_fit, "bounce_velocity", _reference_bounce_velocity),
        (rich_ball_physics, "flight", reference),
        (rich_ball_physics, "bounce_velocity", _reference_bounce_velocity),
    )
    originals = [(module, name, getattr(module, name)) for module, name, _ in replacements]
    for module, name, replacement in replacements:
        setattr(module, name, replacement)
    try:
        yield
    finally:
        for module, name, original in reversed(originals):
            setattr(module, name, original)


def _diagnostic_case(case: FlightCase, synthetic: dict[str, Any]) -> FlightCase:
    result = copy.copy(case)
    result.contacts = copy.deepcopy(synthetic["contacts"])
    result.anchors = copy.deepcopy(synthetic["anchors"])
    result.ball = {frame: pixel.copy() for frame, pixel in synthetic["ball"].items()}
    result.players = copy.deepcopy(synthetic["players"])
    result.observation_weights = {frame: 1.0 for frame in synthetic["ball"]}
    return result


def _fit_one(
    case: FlightCase,
    *,
    reference_model: bool,
    max_nfev: int,
) -> dict[str, Any]:
    context = reference_fitter_adapter() if reference_model else nullcontext()
    with context:
        fit = anchor_first_fit.fit_anchor_first_shot(
            0,
            case.contacts,
            case.ball,
            case.players,
            case.camera,
            case.fps,
            case.surface,
            max_nfev,
            case.anchors,
            observation_weights=case.observation_weights,
        )
        status, reasons, compact = compact_and_classify(case, fit)
    return {
        "flight_id": case.flight_id,
        "match_id": case.match_id,
        "clip": case.clip,
        "surface": case.surface,
        "solved": fit is not None,
        "accepted": status == "provisional_valid",
        "status": status,
        "reasons": reasons,
        "held_out_median_px": (compact.get("held_out_reprojection_median_px") if compact else None),
        "held_out_p90_px": compact.get("held_out_reprojection_p90_px") if compact else None,
        "anchor_max_error_m": compact.get("anchor_max_error_m") if compact else None,
        "optimizer_nfev": compact.get("optimizer_nfev") if compact else None,
    }


def _summarize(arm: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    medians = [row["held_out_median_px"] for row in rows if row["held_out_median_px"] is not None]
    p90s = [row["held_out_p90_px"] for row in rows if row["held_out_p90_px"] is not None]
    return {
        "arm": arm,
        "attempted": len(rows),
        "solved": sum(bool(row["solved"]) for row in rows),
        "accepted": sum(bool(row["accepted"]) for row in rows),
        "held_out_flight_median_px_corpus_median": percentile(medians, 50.0),
        "held_out_flight_median_px_corpus_p90": percentile(medians, 90.0),
        "held_out_flight_p90_px_corpus_median": percentile(p90s, 50.0),
        "held_out_flight_p90_px_corpus_p90": percentile(p90s, 90.0),
    }


def run_closed_loop(
    oracle_root: Path,
    output_root: Path,
    *,
    max_nfev: int = 20,
    limit: int = TARGET_CLOSED_LOOP_FLIGHTS,
) -> dict[str, Any]:
    """Generate once with reference physics, then fit with current and reference laws."""

    rng = np.random.default_rng(SEED)
    selected: list[tuple[FlightCase, dict[str, Any]]] = []
    rejected: list[dict[str, str]] = []
    for case in load_flight_cases(oracle_root, solved_only=True):
        try:
            with reference_fitter_adapter():
                synthetic = s6root_closed_loop.synthesize(case, rng)
        except (ValueError, FloatingPointError) as exc:
            rejected.append({"flight_id": case.flight_id, "reason": str(exc)})
            continue
        selected.append((case, synthetic))
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise RuntimeError(f"only built {len(selected)} of {limit} reference synthetic flights")

    all_rows: dict[str, list[dict[str, Any]]] = {}
    for arm, use_reference in (
        ("reference_generated_current_fit", False),
        ("reference_generated_reference_fit", True),
    ):
        rows = []
        for case, synthetic in selected:
            row = _fit_one(
                _diagnostic_case(case, synthetic),
                reference_model=use_reference,
                max_nfev=max_nfev,
            )
            row["arm"] = arm
            rows.append(row)
        all_rows[arm] = rows
    table = [_summarize(arm, rows) for arm, rows in all_rows.items()]
    payload = {
        "schema": "wk3_physics_reference_closed_loop_v1",
        "artifact_class": "oracle_diagnostic",
        "seed": SEED,
        "noise_sigma_px": 1.0,
        "selection": f"first {limit} lexicographic solved Arm-D flights usable by reference model",
        "generator": "physics.reference",
        "fitter": "cv.pipeline.anchor_first_fit.fit_anchor_first_shot",
        "adapter": "validation-local symbol substitution; pipeline source unchanged",
        "table": table,
        "rows": [row for rows in all_rows.values() for row in rows],
        "selection_rejections": rejected,
    }
    write_json(output_root / "closed_loop.json", payload)
    write_csv(output_root / "closed_loop.csv", table, list(table[0]))
    return payload


def run_oracle_arm_d(
    oracle_root: Path,
    output_root: Path,
    *,
    max_nfev: int = 20,
    limit: int | None = None,
) -> dict[str, Any]:
    """Refit identical owner Arm-D inputs under current and reference physics."""

    cases = load_flight_cases(oracle_root)
    if limit is not None:
        cases = cases[:limit]
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm, use_reference in (("current_physics", False), ("reference_physics", True)):
        rows = []
        for index, case in enumerate(cases, 1):
            rows.append(
                {
                    **_fit_one(case, reference_model=use_reference, max_nfev=max_nfev),
                    "arm": arm,
                }
            )
            if index % 25 == 0 or index == len(cases):
                print(f"oracle {arm}: {index}/{len(cases)}", flush=True)
        rows_by_arm[arm] = rows
    table = [_summarize(arm, rows) for arm, rows in rows_by_arm.items()]
    payload = {
        "schema": "wk3_physics_oracle_arm_d_refit_v1",
        "artifact_class": "oracle_diagnostic",
        "oracle_root": os.fspath(oracle_root),
        "attempt_population": "all flight_attempts in oracle Arm-D report",
        "fitter": "unchanged cv.pipeline.anchor_first_fit.fit_anchor_first_shot",
        "adapter": "validation-local symbol substitution; pipeline source unchanged",
        "max_nfev": max_nfev,
        "table": table,
        "rows": [row for rows in rows_by_arm.values() for row in rows],
    }
    write_json(output_root / "oracle_arm_d.json", payload)
    write_csv(output_root / "oracle_arm_d.csv", table, list(table[0]))
    return payload


def _ground_crossing(
    module: Any,
    speed: float,
    angle_deg: float,
    spin_rpm: float,
    height_m: float,
    *,
    params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    angle = math.radians(angle_deg)
    velocity = np.array([speed * math.cos(angle), 0.0, speed * math.sin(angle)])
    spin = np.array([0.0, spin_rpm * 2.0 * math.pi / 60.0, 0.0])
    kwargs = {} if params is None else {"params": params}
    return module.RK4(
        0.0005,
        10_000,
        np.array([0.0, 0.0, height_m]),
        velocity,
        spin,
        verbose=False,
        **kwargs,
    )


def _height_at_x(positions: np.ndarray, x_m: float) -> float | None:
    index = np.flatnonzero(positions[:, 0] >= x_m)
    if not len(index) or index[0] == 0:
        return None
    upper = int(index[0])
    lower = upper - 1
    fraction = (x_m - positions[lower, 0]) / (positions[upper, 0] - positions[lower, 0])
    return float((1.0 - fraction) * positions[lower, 2] + fraction * positions[upper, 2])


def _solve_current_angle(speed: float, spin_rpm: float, height_m: float, target_m: float) -> float:
    def residual(angle: float) -> float:
        return (
            float(_ground_crossing(current_flight, speed, angle, spin_rpm, height_m)[1][-1, 0])
            - target_m
        )

    grid = np.linspace(-15.0, 35.0, 101)
    values = [residual(float(angle)) for angle in grid]
    brackets = [
        (float(a), float(b))
        for a, b, fa, fb in zip(grid[:-1], grid[1:], values[:-1], values[1:], strict=True)
        if fa == 0.0 or fa * fb < 0.0
    ]
    if not brackets:
        raise RuntimeError("could not solve launch angle for representative trajectory")
    return float(brentq(residual, *brackets[0]))


def trajectory_comparison(output_root: Path) -> dict[str, Any]:
    """Sequentially quantify each flight-model deviation on two named trajectories."""

    production_decay_per_m = (
        current_flight.default_params["C_spin_decay"]
        * 0.5
        * current_flight.default_params["rho_air"]
        * math.pi
        * current_flight.default_params["R_ball"] ** 4
        / current_flight.default_params["J_ball"]
    )
    production_equivalent = {
        "g": current_flight.default_params["g"],
        "m_ball": current_flight.default_params["m_ball"],
        "R_ball": current_flight.default_params["R_ball"],
        "J_ball": current_flight.default_params["J_ball"],
        "rho_air": current_flight.default_params["rho_air"],
        "coefficient_model": "cross_2020",
        "C_drag": current_flight.default_params["C_drag"],
        "C_lift_slope": current_flight.default_params["C_lift"],
        "spin_decay_per_m": production_decay_per_m,
    }
    paper_constants = {
        **production_equivalent,
        "R_ball": reference.R_BALL,
        "J_ball": reference.J_BALL,
        "rho_air": reference.RHO_AIR,
    }
    empirical_drag = {key: value for key, value in paper_constants.items() if key != "C_drag"}
    empirical_drag.update({"drag_model": "stepanek", "lift_model": "cross_2020"})
    saturated_lift = {**empirical_drag, "lift_model": "stepanek"}
    measured_decay = {
        **saturated_lift,
        "spin_decay_per_m": reference.SPIN_DECAY_PER_M,
    }
    profiles: list[tuple[str, Any, dict[str, Any] | None]] = [
        ("current_code", current_flight, None),
        ("reference_engine_current_equations", reference, production_equivalent),
        ("paper_radius_density", reference, paper_constants),
        ("stepanek_spin_dependent_drag", reference, empirical_drag),
        ("stepanek_saturated_lift", reference, saturated_lift),
        ("measured_spin_decay", reference, measured_decay),
    ]
    cases = (
        {
            "shot": "groundstroke",
            "speed_ms": 30.0,
            "spin_rpm": 2500.0,
            "contact_height_m": 1.0,
            "target_landing_m": 22.77,
        },
        {
            "shot": "serve",
            "speed_ms": 55.0,
            "spin_rpm": 2000.0,
            "contact_height_m": 2.8,
            "target_landing_m": 18.285,
        },
    )
    rows: list[dict[str, Any]] = []
    for case in cases:
        angle = _solve_current_angle(
            case["speed_ms"],
            case["spin_rpm"],
            case["contact_height_m"],
            case["target_landing_m"],
        )
        previous: dict[str, float | None] | None = None
        for profile_name, module, params in profiles:
            _, positions, velocities, spins = _ground_crossing(
                module,
                case["speed_ms"],
                angle,
                case["spin_rpm"],
                case["contact_height_m"],
                params=params,
            )
            landing = float(positions[-1, 0])
            net_height = _height_at_x(positions, 11.885)
            record = {
                **case,
                "launch_angle_deg": angle,
                "profile": profile_name,
                "landing_m": landing,
                "net_height_m": net_height,
                "landing_increment_m": None
                if previous is None
                else landing - previous["landing_m"],
                "net_height_increment_m": (
                    None
                    if previous is None or net_height is None or previous["net_height_m"] is None
                    else net_height - previous["net_height_m"]
                ),
                "landing_speed_ms": float(np.linalg.norm(velocities[-1])),
                "landing_spin_rpm": float(np.linalg.norm(spins[-1])) * 60.0 / (2.0 * math.pi),
            }
            rows.append(record)
            previous = {"landing_m": landing, "net_height_m": net_height}

    bounce_rows = []
    for case in cases:
        final = next(
            row
            for row in rows
            if row["shot"] == case["shot"] and row["profile"] == "measured_spin_decay"
        )
        angle = float(final["launch_angle_deg"])
        _, _, velocities, spins = _ground_crossing(
            reference,
            case["speed_ms"],
            angle,
            case["spin_rpm"],
            case["contact_height_m"],
            params=measured_decay,
        )
        incoming_velocity = velocities[-1]
        incoming_spin = spins[-1]
        current = current_impact.court_bounce_vector(
            incoming_velocity, incoming_spin, surface="hard"
        )
        faithful = reference.court_bounce(incoming_velocity, incoming_spin, "hard")
        for name, velocity, spin, regime in (
            ("current_bounce", current.velocity, current.spin, current.regime),
            ("reference_bounce", faithful.velocity, faithful.spin, faithful.regime),
        ):
            bounce_rows.append(
                {
                    "shot": case["shot"],
                    "model": name,
                    "outgoing_speed_ms": float(np.linalg.norm(velocity)),
                    "outgoing_vertical_ms": float(velocity[2]),
                    "outgoing_spin_rpm": float(np.linalg.norm(spin)) * 60.0 / (2.0 * math.pi),
                    "regime": regime,
                }
            )
    payload = {
        "schema": "wk3_physics_trajectory_comparison_v1",
        "method": "launch angle chosen so current physics lands at named target; identical initial state in every profile",
        "net_plane_distance_m": 11.885,
        "rows": rows,
        "bounce_rows": bounce_rows,
    }
    write_json(output_root / "trajectory_comparison.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = data_root()
    parser.add_argument("--oracle-root", type=Path, default=root / ORACLE_RELATIVE)
    parser.add_argument("--output-root", type=Path, default=root / OUTPUT_RELATIVE)
    parser.add_argument("--max-nfev", type=int, default=20)
    parser.add_argument("--closed-loop-limit", type=int, default=TARGET_CLOSED_LOOP_FLIGHTS)
    parser.add_argument("--oracle-limit", type=int)
    parser.add_argument(
        "--stage",
        choices=("all", "comparison", "closed-loop", "oracle"),
        default="all",
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema": "wk3_physics_validation_summary_v1",
        "output_root": os.fspath(args.output_root),
    }
    existing_outputs = {
        "trajectory_comparison": ("trajectory_comparison.json", None),
        "closed_loop": ("closed_loop.json", "table"),
        "oracle_arm_d": ("oracle_arm_d.json", "table"),
    }
    for name, (filename, key) in existing_outputs.items():
        path = args.output_root / filename
        if path.exists():
            payload = json.loads(path.read_text())
            summary[name] = payload if key is None else payload[key]
    if args.stage in {"all", "comparison"}:
        summary["trajectory_comparison"] = trajectory_comparison(args.output_root)
    if args.stage in {"all", "closed-loop"}:
        summary["closed_loop"] = run_closed_loop(
            args.oracle_root,
            args.output_root,
            max_nfev=args.max_nfev,
            limit=args.closed_loop_limit,
        )["table"]
    if args.stage in {"all", "oracle"}:
        summary["oracle_arm_d"] = run_oracle_arm_d(
            args.oracle_root,
            args.output_root,
            max_nfev=args.max_nfev,
            limit=args.oracle_limit,
        )["table"]
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
