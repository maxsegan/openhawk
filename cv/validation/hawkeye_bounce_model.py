"""Measure the court bounce from consecutive Hawk-Eye strike pairs.

Evaluation-only.  For every consecutive strike pair in the Hawk-Eye CourtVision corpus
(``data/external/ryurko-hawkeye``) this module fits the incoming flight of
strike ``k`` and the outgoing flight of the same bounce with :mod:`physics.flight`, then
reports the measured velocity change at the court impact: vertical restitution, horizontal
speed retention, and the skid/grip regime, split by surface and by incoming speed and angle.

The corpus has no timestamps.  Both flights are therefore identified from position landmarks
only: ``hit``, ``peak`` (pre-bounce apex), ``net`` and ``bounce`` for the incoming arc, and
``bounce``, ``peak`` (post-bounce apex) and the next strike's ``hit`` for the outgoing arc.

Two populations exist and they carry different evidence:

``pair``
    A rally bounce with the next strike's hit.  The outgoing velocity is fully identified.
    Incoming spin is *not* measured for these strikes; the play-by-play ``spin_rpm`` column
    is populated only for the last strike of a point.
``terminal``
    The last strike of a point.  Incoming spin is measured, but there is no next hit, so the
    outgoing arc is identified from the post-bounce apex alone.

Outgoing spin is reported as measured only if the residual profile shows it is identifiable.
It is not: :func:`spin_identifiability_profile` measures a flat residual over 0-4000 rpm.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import least_squares

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cv.validation import hawkeye_flight_fit as hff  # noqa: E402
from physics import flight, impact, reference  # noqa: E402

CORPUS_ROOT = Path("data/external/ryurko-hawkeye")
AERO_FIT_RELATIVE = Path("data/processed/external/hawkeye_flight_fit_v3.json")

RPM_PER_RADSEC = 60.0 / (2.0 * math.pi)
RADSEC_PER_RPM = 1.0 / RPM_PER_RADSEC

# The v3 aero study leaves both free-flight coefficients essentially unconstrained by this
# corpus: its apex residual moves by ~6 mm across C_lift in [0.05, 1.2] and by ~2 mm across
# C_spin_decay in [0, 0.2], and C_drag is absorbed by the free launch speed.  The measurement
# therefore uses the production constants the Stage-6 fitter itself integrates with, so that
# no aero mismatch is folded into the bounce numbers.
AERO_PARAMS = dict(flight.default_params)

MAX_FLIGHT_SECONDS = 2.5
SAMPLE_DT = 0.005
SUBSTEPS_PER_SECOND = 200.0

# Fit-quality gates.  Both are position residual RMS in metres over the landmark set.
MAX_INCOMING_RMSE_M = 0.20
MAX_OUTGOING_RMSE_M = 0.10

SPEED_EDGES_MS = (0.0, 20.0, 25.0, 30.0, math.inf)
ANGLE_EDGES_DEG = (0.0, 12.0, 16.0, 20.0, 90.0)
SPIN_PROFILE_RPM = (0.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return {
            "n": 0,
            "median": None,
            "mean": None,
            "q05": None,
            "q25": None,
            "q75": None,
            "q95": None,
            "std": None,
        }
    array = np.asarray(finite, dtype=float)
    return {
        "n": int(array.size),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "std": float(np.std(array)),
    }


def topspin_axis(velocity: Sequence[float]) -> np.ndarray:
    """Unit spin axis whose Magnus force pushes a forward-moving ball down."""
    horizontal = np.array([float(velocity[0]), float(velocity[1]), 0.0])
    norm = float(np.linalg.norm(horizontal))
    if norm < 1e-9:
        return np.zeros(3)
    horizontal /= norm
    return np.array([-horizontal[1], horizontal[0], 0.0])


def _dense(
    start: Sequence[float],
    velocity: Sequence[float],
    spin: Sequence[float],
    *,
    max_seconds: float = MAX_FLIGHT_SECONDS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times = np.arange(0.0, max_seconds + 1e-9, SAMPLE_DT)
    positions, velocities, spins = flight.sample_states(
        np.asarray(start, dtype=float),
        np.asarray(velocity, dtype=float),
        np.asarray(spin, dtype=float),
        times,
        substeps_per_second=SUBSTEPS_PER_SECOND,
        params=AERO_PARAMS,
    )
    return times, positions, velocities, spins


def _apex(positions: np.ndarray, velocities: np.ndarray) -> np.ndarray | None:
    vertical = velocities[:, 2]
    crossings = np.flatnonzero((vertical[:-1] > 0.0) & (vertical[1:] <= 0.0))
    if not len(crossings):
        return None
    index = int(crossings[0])
    fraction = float(vertical[index] / (vertical[index] - vertical[index + 1]))
    return positions[index] + fraction * (positions[index + 1] - positions[index])


def _at_horizontal_distance(
    positions: np.ndarray, origin: Sequence[float], distance: float
) -> np.ndarray | None:
    travelled = np.hypot(positions[:, 0] - origin[0], positions[:, 1] - origin[1])
    if float(travelled[-1]) < distance:
        return None
    index = int(np.searchsorted(travelled, distance))
    index = max(1, min(index, len(travelled) - 1))
    span = max(float(travelled[index] - travelled[index - 1]), 1e-9)
    fraction = (distance - float(travelled[index - 1])) / span
    return positions[index - 1] + fraction * (positions[index] - positions[index - 1])


def _plane_crossing(positions: np.ndarray, axis: int, value: float) -> np.ndarray | None:
    offsets = positions[:, axis] - value
    crossings = np.flatnonzero(np.sign(offsets[:-1]) * np.sign(offsets[1:]) < 0.0)
    if not len(crossings):
        return None
    index = int(crossings[0])
    span = float(offsets[index] - offsets[index + 1])
    if abs(span) < 1e-12:
        return positions[index]
    fraction = float(offsets[index]) / span
    return positions[index] + fraction * (positions[index + 1] - positions[index])


def _ground_state(
    start: Sequence[float],
    velocity: Sequence[float],
    spin: Sequence[float],
    target_z: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Integrate to the first descending crossing of ``target_z``."""
    times, positions, velocities, spins = _dense(start, velocity, spin)
    offsets = positions[:, 2] - target_z
    descending = np.flatnonzero((offsets[:-1] > 0.0) & (offsets[1:] <= 0.0))
    if not len(descending):
        return None
    index = int(descending[0])
    fraction = float(offsets[index] / (offsets[index] - offsets[index + 1]))

    def blend(values: np.ndarray) -> np.ndarray:
        return values[index] + fraction * (values[index + 1] - values[index])

    return blend(positions), blend(velocities), blend(spins)


@dataclass(frozen=True)
class BouncePair:
    """One measured court impact plus the flights that bracket it."""

    match_file: str
    point_id: str
    serve_num: str
    strike_index: int
    surface: str
    tour: str
    population: str
    hit: tuple[float, float, float]
    peak_in: tuple[float, float, float] | None
    net: tuple[float, float, float] | None
    bounce: tuple[float, float, float]
    peak_out: tuple[float, float, float]
    next_hit: tuple[float, float, float] | None
    measured_spin_rpm: float | None


def enumerate_pairs(traj_path: str, pbp_dir: str) -> list[BouncePair]:
    """Return every usable court impact of one match, in both populations."""
    strikes = hff.load_strikes(traj_path, pbp_dir)
    grouped: dict[tuple[str, str], list[hff.Strike]] = defaultdict(list)
    for strike in strikes:
        grouped[(strike.key[1], strike.key[2])].append(strike)
    output: list[BouncePair] = []
    for (point_id, serve_num), rally in grouped.items():
        rally.sort(key=lambda row: row.strike_index)
        following = {row.strike_index: row for row in rally}
        for strike in rally:
            if strike.peak_out is None or strike.bounce is None:
                continue
            successor = following.get(strike.strike_index + 1)
            if successor is not None:
                population = "pair"
                next_hit = successor.hit
            elif strike.is_last:
                population = "terminal"
                next_hit = None
            else:
                continue
            output.append(
                BouncePair(
                    match_file=os.path.basename(traj_path),
                    point_id=point_id,
                    serve_num=serve_num,
                    strike_index=int(strike.strike_index),
                    surface=strike.surface,
                    tour=strike.tour,
                    population=population,
                    hit=tuple(strike.hit),
                    peak_in=tuple(strike.peak_in) if strike.peak_in else None,
                    net=tuple(strike.net) if strike.net else None,
                    bounce=tuple(strike.bounce),
                    peak_out=tuple(strike.peak_out),
                    next_hit=tuple(next_hit) if next_hit else None,
                    measured_spin_rpm=strike.spin_rpm,
                )
            )
    return output


def solve_incoming(pair: BouncePair, spin_rpm: float) -> dict[str, Any] | None:
    """Recover the incoming velocity and spin at the court impact.

    Free parameters are launch speed, elevation and yaw.  Spin is held fixed: the v3 study
    established that per-shot spin is not identifiable from these landmarks, so leaving it
    free would only absorb residual without measuring anything.
    """
    if pair.net is None:
        return None
    hit = np.asarray(pair.hit, dtype=float)
    bounce = np.asarray(pair.bounce, dtype=float)
    heading = math.atan2(bounce[1] - hit[1], bounce[0] - hit[0])
    spin_magnitude = float(spin_rpm) * RADSEC_PER_RPM
    apex_z = float(pair.peak_in[2]) if pair.peak_in is not None else float(hit[2])
    horizontal = math.hypot(bounce[0] - hit[0], bounce[1] - hit[1])
    rise = max(apex_z - float(hit[2]), 0.0)
    up = math.sqrt(2.0 * rise / AERO_PARAMS["g"]) if rise > 0.0 else 0.0
    down = math.sqrt(2.0 * max(apex_z, 0.05) / AERO_PARAMS["g"])
    guess_speed = horizontal / max(up + down, 0.25)

    def residual(parameters: np.ndarray) -> list[float]:
        speed, elevation, yaw = parameters
        direction = heading + yaw
        velocity = np.array(
            [
                speed * math.cos(elevation) * math.cos(direction),
                speed * math.cos(elevation) * math.sin(direction),
                speed * math.sin(elevation),
            ]
        )
        spin = spin_magnitude * topspin_axis(velocity)
        _, positions, velocities, _ = _dense(hit, velocity, spin)
        net_point = _plane_crossing(positions, 0, float(pair.net[0]))
        apex = _apex(positions, velocities)
        ground = _ground_state(hit, velocity, spin, float(bounce[2]))
        if net_point is None or apex is None or ground is None:
            return [25.0] * 5
        return [
            float(net_point[1] - pair.net[1]),
            float(net_point[2] - pair.net[2]),
            float(ground[0][0] - bounce[0]),
            float(ground[0][1] - bounce[1]),
            float(apex[2] - apex_z),
        ]

    try:
        solution = least_squares(
            residual,
            [max(guess_speed, 10.0), math.radians(8.0), 0.0],
            bounds=(
                [5.0, math.radians(-30.0), math.radians(-30.0)],
                [65.0, math.radians(60.0), math.radians(30.0)],
            ),
            xtol=1e-4,
            ftol=1e-4,
            max_nfev=60,
        )
    except (ValueError, np.linalg.LinAlgError):
        return None
    values = np.asarray(residual(solution.x), dtype=float)
    rmse = float(np.sqrt(np.mean(values**2)))
    if not math.isfinite(rmse) or rmse > MAX_INCOMING_RMSE_M:
        return None
    speed, elevation, yaw = solution.x
    direction = heading + yaw
    velocity = np.array(
        [
            speed * math.cos(elevation) * math.cos(direction),
            speed * math.cos(elevation) * math.sin(direction),
            speed * math.sin(elevation),
        ]
    )
    spin = spin_magnitude * topspin_axis(velocity)
    ground = _ground_state(hit, velocity, spin, float(bounce[2]))
    if ground is None:
        return None
    position, impact_velocity, impact_spin = ground
    return {
        "rmse_m": rmse,
        "launch_speed_ms": float(speed),
        "launch_elevation_deg": math.degrees(float(elevation)),
        "impact_position": position.tolist(),
        "impact_velocity": impact_velocity.tolist(),
        "impact_spin": impact_spin.tolist(),
        "assumed_spin_rpm": float(spin_rpm),
    }


def solve_outgoing(pair: BouncePair, spin_rpm: float = 0.0) -> dict[str, Any] | None:
    """Recover the outgoing velocity at the court impact.

    The rebound arc lasts well under a second, so the spin term is a negligible force there;
    :func:`spin_identifiability_profile` measures that directly.  Spin is therefore fixed and
    only the three outgoing velocity components are free.
    """
    bounce = np.asarray(pair.bounce, dtype=float)
    peak_out = np.asarray(pair.peak_out, dtype=float)
    spin_magnitude = float(spin_rpm) * RADSEC_PER_RPM
    rise = float(peak_out[2] - bounce[2])
    if rise <= 0.0:
        return None
    vertical = math.sqrt(2.0 * AERO_PARAMS["g"] * rise)
    time_up = max(vertical / AERO_PARAMS["g"], 1e-3)
    guess = [
        float(peak_out[0] - bounce[0]) / time_up,
        float(peak_out[1] - bounce[1]) / time_up,
        vertical,
    ]
    next_hit = np.asarray(pair.next_hit, dtype=float) if pair.next_hit is not None else None
    hit_distance = (
        float(np.hypot(next_hit[0] - bounce[0], next_hit[1] - bounce[1]))
        if next_hit is not None
        else None
    )

    def residual(parameters: np.ndarray) -> list[float]:
        velocity = np.asarray(parameters, dtype=float)
        spin = spin_magnitude * topspin_axis(velocity)
        _, positions, velocities, _ = _dense(bounce, velocity, spin)
        apex = _apex(positions, velocities)
        if apex is None:
            return [25.0] * (5 if next_hit is not None else 3)
        rows = [
            float(apex[0] - peak_out[0]),
            float(apex[1] - peak_out[1]),
            float(apex[2] - peak_out[2]),
        ]
        if next_hit is None:
            return rows
        arrival = _at_horizontal_distance(positions, bounce, hit_distance)
        if arrival is None:
            return [25.0] * 5
        rows.append(float(math.hypot(arrival[0] - next_hit[0], arrival[1] - next_hit[1])))
        rows.append(float(arrival[2] - next_hit[2]))
        return rows

    try:
        solution = least_squares(
            residual,
            guess,
            bounds=([-60.0, -60.0, 0.2], [60.0, 60.0, 25.0]),
            xtol=1e-4,
            ftol=1e-4,
            max_nfev=60,
        )
    except (ValueError, np.linalg.LinAlgError):
        return None
    values = np.asarray(residual(solution.x), dtype=float)
    rmse = float(np.sqrt(np.mean(values**2)))
    if not math.isfinite(rmse) or rmse > MAX_OUTGOING_RMSE_M:
        return None
    return {
        "rmse_m": rmse,
        "outgoing_velocity": [float(value) for value in solution.x],
        "assumed_spin_rpm": float(spin_rpm),
    }


def spin_identifiability_profile(
    pair: BouncePair, incoming: dict[str, Any]
) -> list[dict[str, float]] | None:
    """Refit the outgoing velocity at fixed spins and report the residual curve.

    The measured quantities are recomputed at every grid point so the curve states directly
    how much of the restitution and retention estimate rests on the assumed outgoing spin.
    """
    velocity_in = np.asarray(incoming["impact_velocity"], dtype=float)
    horizontal_in = float(math.hypot(velocity_in[0], velocity_in[1]))
    vertical_in = float(abs(velocity_in[2]))
    if horizontal_in < 2.0 or vertical_in < 1.0:
        return None
    curve = []
    for rpm in SPIN_PROFILE_RPM:
        solved = solve_outgoing(pair, spin_rpm=rpm)
        if solved is None:
            return None
        velocity_out = np.asarray(solved["outgoing_velocity"], dtype=float)
        curve.append(
            {
                "spin_rpm": float(rpm),
                "rmse_m": float(solved["rmse_m"]),
                "outgoing_speed_ms": float(np.linalg.norm(velocity_out)),
                "restitution": float(velocity_out[2] / vertical_in),
                "horizontal_retention": float(
                    math.hypot(velocity_out[0], velocity_out[1]) / horizontal_in
                ),
            }
        )
    return curve


def cross_prediction(
    horizontal_in: float, vertical_in: float, topspin_radps: float, surface: str
) -> dict[str, float] | None:
    """Cross bounce prediction from the two implementations in the repository."""
    try:
        shipped = impact.court_bounce(horizontal_in, vertical_in, topspin_radps, surface=surface)
    except (ValueError, ZeroDivisionError):
        return None
    incidence = math.degrees(math.atan2(vertical_in, max(horizontal_in, 1e-9)))
    try:
        reference_result = reference.court_bounce(
            np.array([horizontal_in, 0.0, -vertical_in]),
            np.array([0.0, topspin_radps, 0.0]),
            surface=surface,
        )
        reference_horizontal = float(math.hypot(*reference_result.velocity[:2]))
        reference_vertical = float(reference_result.velocity[2])
    except (ValueError, ZeroDivisionError):
        reference_horizontal = math.nan
        reference_vertical = math.nan
    return {
        "incidence_deg": incidence,
        "e_y_shipped": float(shipped.vy2 / max(vertical_in, 1e-9)),
        "horizontal_retention_shipped": float(shipped.vx2 / max(horizontal_in, 1e-9)),
        "outgoing_spin_rpm_shipped": float(shipped.w2 * RPM_PER_RADSEC),
        "regime_shipped": str(shipped.regime),
        "e_y_reference": reference_vertical / max(vertical_in, 1e-9),
        "horizontal_retention_reference": reference_horizontal / max(horizontal_in, 1e-9),
    }


def measure_pair(pair: BouncePair, incoming_spin_rpm: float) -> dict[str, Any] | None:
    """Full measurement of one court impact.

    The outgoing arc's own spin is not identifiable from its landmarks, so the rebound solve
    is run at the Cross model's predicted outgoing topspin.  That choice is stated in every
    row (``assumed_outgoing_spin_rpm``) and its cost is measured directly by
    :func:`spin_identifiability_profile`; it does not affect the incoming solve at all.
    """
    incoming = solve_incoming(pair, incoming_spin_rpm)
    if incoming is None:
        return None
    velocity_in = np.asarray(incoming["impact_velocity"], dtype=float)
    horizontal_in = float(math.hypot(velocity_in[0], velocity_in[1]))
    vertical_in = float(abs(velocity_in[2]))
    if vertical_in < 1.0 or horizontal_in < 2.0:
        return None
    spin_in = np.asarray(incoming["impact_spin"], dtype=float)
    topspin_in = float(np.dot(spin_in, topspin_axis(velocity_in)))
    prediction = cross_prediction(horizontal_in, vertical_in, topspin_in, pair.surface)
    if prediction is None:
        return None
    outgoing = solve_outgoing(pair, spin_rpm=prediction["outgoing_spin_rpm_shipped"])
    if outgoing is None:
        return None
    # The same rebound solved with no outgoing spin at all.  The corpus cannot distinguish
    # the two; reporting both endpoints is the only honest statement of what it measures.
    spin_free = solve_outgoing(pair, spin_rpm=0.0)
    velocity_out = np.asarray(outgoing["outgoing_velocity"], dtype=float)
    horizontal_out = float(math.hypot(velocity_out[0], velocity_out[1]))
    vertical_out = float(velocity_out[2])
    restitution = vertical_out / vertical_in
    retention = horizontal_out / horizontal_in
    if not (0.05 < restitution < 1.3) or not (0.1 < retention < 1.3):
        return None
    # Cross's own regime test at separation: the ball slid throughout the impact when the
    # contact point is still moving forwards, i.e. R*w_out < v_horizontal_out.
    rolling_spin = horizontal_out / AERO_PARAMS["R_ball"]
    incidence = math.degrees(math.atan2(vertical_in, horizontal_in))
    heading = velocity_in[:2] / max(float(np.linalg.norm(velocity_in[:2])), 1e-9)
    deflection = math.degrees(
        math.atan2(
            float(velocity_out[0] * heading[1] - velocity_out[1] * heading[0]),
            float(velocity_out[0] * heading[0] + velocity_out[1] * heading[1]),
        )
    )
    return {
        "match_file": pair.match_file,
        "point_id": pair.point_id,
        "serve_num": pair.serve_num,
        "strike_index": pair.strike_index,
        "surface": pair.surface,
        "tour": pair.tour,
        "population": pair.population,
        "incoming_rmse_m": incoming["rmse_m"],
        "outgoing_rmse_m": outgoing["rmse_m"],
        "incoming_spin_rpm": incoming["assumed_spin_rpm"],
        "incoming_spin_measured": pair.measured_spin_rpm is not None,
        "assumed_outgoing_spin_rpm": outgoing["assumed_spin_rpm"],
        "horizontal_in_ms": horizontal_in,
        "vertical_in_ms": vertical_in,
        "speed_in_ms": float(np.linalg.norm(velocity_in)),
        "incidence_deg": incidence,
        "horizontal_out_ms": horizontal_out,
        "vertical_out_ms": vertical_out,
        "restitution": restitution,
        "horizontal_retention": retention,
        "deflection_deg": deflection,
        "rolling_spin_rpm": rolling_spin * RPM_PER_RADSEC,
        "restitution_spin_free": (
            float(spin_free["outgoing_velocity"][2] / vertical_in)
            if spin_free is not None
            else None
        ),
        "horizontal_retention_spin_free": (
            float(
                math.hypot(spin_free["outgoing_velocity"][0], spin_free["outgoing_velocity"][1])
                / horizontal_in
            )
            if spin_free is not None
            else None
        ),
        "bounce_x": float(pair.bounce[0]),
        "bounce_y": float(pair.bounce[1]),
        "apex_out_height_m": float(pair.peak_out[2]),
        **{f"cross_{key}": value for key, value in prediction.items()},
    }


_WORKER_SPIN: dict[str, float] = {}


def _init_worker(spin_by_surface: dict[str, float]) -> None:
    global _WORKER_SPIN
    _WORKER_SPIN = spin_by_surface


def _measure_worker(pair: BouncePair) -> dict[str, Any] | None:
    spin = (
        float(pair.measured_spin_rpm)
        if pair.measured_spin_rpm is not None
        else _WORKER_SPIN.get(pair.surface, 1900.0)
    )
    try:
        return measure_pair(pair, spin)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return None


def _bucket(value: float, edges: Sequence[float]) -> str:
    for low, high in zip(edges, edges[1:]):
        if low <= value < high:
            high_label = "inf" if math.isinf(high) else f"{high:g}"
            return f"{low:g}-{high_label}"
    return "out_of_range"


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the per-bounce measurements into the reported distributions."""

    def block(subset: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not subset:
            return {"n": 0}
        return {
            "n": len(subset),
            "restitution": _quantiles([row["restitution"] for row in subset]),
            "restitution_spin_free": _quantiles([row["restitution_spin_free"] for row in subset]),
            "horizontal_retention_spin_free": _quantiles(
                [row["horizontal_retention_spin_free"] for row in subset]
            ),
            "horizontal_retention": _quantiles([row["horizontal_retention"] for row in subset]),
            "incidence_deg": _quantiles([row["incidence_deg"] for row in subset]),
            "speed_in_ms": _quantiles([row["speed_in_ms"] for row in subset]),
            "vertical_in_ms": _quantiles([row["vertical_in_ms"] for row in subset]),
            "deflection_deg": _quantiles([row["deflection_deg"] for row in subset]),
            "apex_out_height_m": _quantiles([row["apex_out_height_m"] for row in subset]),
            "cross_e_y_shipped": _quantiles([row["cross_e_y_shipped"] for row in subset]),
            "cross_horizontal_retention_shipped": _quantiles(
                [row["cross_horizontal_retention_shipped"] for row in subset]
            ),
            "restitution_gap_vs_cross": _quantiles(
                [row["restitution"] - row["cross_e_y_shipped"] for row in subset]
            ),
            "retention_gap_vs_cross": _quantiles(
                [
                    row["horizontal_retention"] - row["cross_horizontal_retention_shipped"]
                    for row in subset
                ]
            ),
            "cross_regime_counts": {
                regime: sum(1 for row in subset if row["regime_shipped_key"] == regime)
                for regime in sorted({row["regime_shipped_key"] for row in subset})
            },
        }

    for row in rows:
        row["regime_shipped_key"] = row.get("cross_regime_shipped", "unknown")

    output: dict[str, Any] = {"all": block(rows)}
    output["by_population"] = {
        population: block([row for row in rows if row["population"] == population])
        for population in sorted({row["population"] for row in rows})
    }
    output["by_surface"] = {}
    for surface in sorted({row["surface"] for row in rows}):
        subset = [row for row in rows if row["surface"] == surface]
        entry = block(subset)
        entry["by_incoming_speed_ms"] = {
            bucket: block(
                [row for row in subset if _bucket(row["speed_in_ms"], SPEED_EDGES_MS) == bucket]
            )
            for bucket in sorted({_bucket(row["speed_in_ms"], SPEED_EDGES_MS) for row in subset})
        }
        entry["by_incidence_deg"] = {
            bucket: block(
                [row for row in subset if _bucket(row["incidence_deg"], ANGLE_EDGES_DEG) == bucket]
            )
            for bucket in sorted({_bucket(row["incidence_deg"], ANGLE_EDGES_DEG) for row in subset})
        }
        entry["by_population"] = {
            population: block([row for row in subset if row["population"] == population])
            for population in sorted({row["population"] for row in subset})
        }
        output["by_surface"][surface] = entry
    return output


def fit_surface_model(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Least-squares surface coefficients for the measured bounce model.

    ``restitution`` is fitted as ``e0 + e1 * (theta1_deg - 16)`` and horizontal retention as
    ``r0 + r1 * (theta1_deg - 16)``, both centred on the corpus median incidence so the
    intercept is the value at a typical groundstroke bounce.  Each quantity is fitted twice:
    at the Cross-predicted outgoing spin (``at_model_spin``) and with a spin-free rebound
    (``spin_free``).  The corpus cannot choose between them, so both are published.
    """
    variants = {
        "at_model_spin": ("restitution", "horizontal_retention"),
        "spin_free": ("restitution_spin_free", "horizontal_retention_spin_free"),
    }
    output: dict[str, Any] = {}
    for surface in sorted({row["surface"] for row in rows}):
        subset = [row for row in rows if row["surface"] == surface]
        if len(subset) < 50:
            continue
        entry: dict[str, Any] = {"n": len(subset)}
        entry["incidence_deg_median"] = float(np.median([row["incidence_deg"] for row in subset]))
        for variant, (restitution_key, retention_key) in variants.items():
            usable = [
                row
                for row in subset
                if row.get(restitution_key) is not None and row.get(retention_key) is not None
            ]
            if len(usable) < 50:
                continue
            angle = np.asarray([row["incidence_deg"] for row in usable], dtype=float) - 16.0
            design = np.stack([np.ones_like(angle), angle], axis=1)
            restitution = np.asarray([row[restitution_key] for row in usable], dtype=float)
            retention = np.asarray([row[retention_key] for row in usable], dtype=float)
            e_coefficients, *_ = np.linalg.lstsq(design, restitution, rcond=None)
            r_coefficients, *_ = np.linalg.lstsq(design, retention, rcond=None)
            entry[variant] = {
                "n": len(usable),
                "restitution_intercept": float(e_coefficients[0]),
                "restitution_slope_per_deg": float(e_coefficients[1]),
                "restitution_residual_std": float(np.std(restitution - design @ e_coefficients)),
                "restitution_median": float(np.median(restitution)),
                "retention_intercept": float(r_coefficients[0]),
                "retention_slope_per_deg": float(r_coefficients[1]),
                "retention_residual_std": float(np.std(retention - design @ r_coefficients)),
                "retention_median": float(np.median(retention)),
            }
        output[surface] = entry
    return output


def corpus_files(root: Path, limit: int) -> list[str]:
    """Deterministic balanced file list across both tournaments."""
    clay = sorted(glob.glob(str(root / "ball_trajectory" / "*roland_garros*.csv")))
    hard = sorted(glob.glob(str(root / "ball_trajectory" / "*australian_open*.csv")))
    if limit <= 0:
        return clay + hard
    half = max(limit // 2, 1)
    return clay[:half] + hard[: limit - min(half, len(clay))]


def measured_spin_priors(pairs: Sequence[BouncePair]) -> dict[str, float]:
    """Median measured incoming topspin per surface, from the terminal population."""
    output: dict[str, float] = {}
    for surface in sorted({pair.surface for pair in pairs}):
        values = [
            float(pair.measured_spin_rpm)
            for pair in pairs
            if pair.surface == surface and pair.measured_spin_rpm is not None
        ]
        if values:
            output[surface] = float(np.median(values))
    return output


def run(
    *,
    corpus_root: Path,
    output_root: Path,
    workers: int,
    max_files: int,
    max_pairs: int,
    profile_samples: int,
) -> dict[str, Any]:
    files = corpus_files(corpus_root, max_files)
    pbp_dir = str(corpus_root / "play_by_play")
    pairs: list[BouncePair] = []
    for path in files:
        pairs.extend(enumerate_pairs(path, pbp_dir))
    priors = measured_spin_priors(pairs)
    rng = np.random.default_rng(20260903)
    if max_pairs > 0 and len(pairs) > max_pairs:
        # Stratify by surface and population so neither slice is starved.
        selected: list[BouncePair] = []
        strata = defaultdict(list)
        for pair in pairs:
            strata[(pair.surface, pair.population)].append(pair)
        per_stratum = max(max_pairs // max(len(strata), 1), 1)
        for key in sorted(strata):
            rows = strata[key]
            if len(rows) > per_stratum:
                index = rng.choice(len(rows), size=per_stratum, replace=False)
                rows = [rows[int(value)] for value in sorted(index)]
            selected.extend(rows)
        pairs = selected
    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=(priors,)
        ) as pool:
            measured = list(pool.map(_measure_worker, pairs, chunksize=8))
    else:
        _init_worker(priors)
        measured = [_measure_worker(pair) for pair in pairs]
    rows = [row for row in measured if row is not None]

    profile_pairs = [pair for pair in pairs if pair.population == "pair"][:profile_samples]
    profiles = []
    for pair in profile_pairs:
        incoming = solve_incoming(pair, priors.get(pair.surface, 1900.0))
        if incoming is None:
            continue
        curve = spin_identifiability_profile(pair, incoming)
        if curve is not None:
            profiles.append({"point_id": pair.point_id, "curve": curve})

    def spread(field: str) -> list[float]:
        return [
            max(row[field] for row in entry["curve"]) - min(row[field] for row in entry["curve"])
            for entry in profiles
        ]

    profile_spread = spread("rmse_m")
    speed_spread = spread("outgoing_speed_ms")

    payload = {
        "schema": "tennis.hawkeye-bounce-model.v1",
        "corpus_root": str(corpus_root),
        "files_read": len(files),
        "pairs_enumerated": len(pairs),
        "pairs_measured": len(rows),
        "aero_params": AERO_PARAMS,
        "aero_provenance": {
            "file": str(AERO_FIT_RELATIVE),
            "sha256": _sha256(REPO_ROOT / AERO_FIT_RELATIVE),
            "note": (
                "v3 textbook_constants; its own C_lift and C_spin_decay grids move the apex "
                "residual by millimetres, so the corpus does not select a different value"
            ),
        },
        "incoming_spin_prior_rpm": priors,
        "fit_gates": {
            "max_incoming_rmse_m": MAX_INCOMING_RMSE_M,
            "max_outgoing_rmse_m": MAX_OUTGOING_RMSE_M,
        },
        "distributions": summarize(rows),
        "surface_model": fit_surface_model(rows),
        "outgoing_spin_identifiability": {
            "design": (
                "refit the outgoing velocity with topspin fixed on a 0-4000 rpm grid and "
                "compare the landmark residual"
            ),
            "samples": len(profiles),
            "rmse_spread_m": _quantiles(profile_spread),
            "outgoing_speed_spread_ms": _quantiles(speed_spread),
            "restitution_spread": _quantiles(spread("restitution")),
            "horizontal_retention_spread": _quantiles(spread("horizontal_retention")),
            "curves": profiles[:20],
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "bounce_model.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    if rows:
        fields = sorted({key for row in rows for key in row})
        with (output_root / "bounces.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=CORPUS_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-files", type=int, default=60)
    parser.add_argument("--max-pairs", type=int, default=6000)
    parser.add_argument("--profile-samples", type=int, default=40)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    payload = run(
        corpus_root=args.corpus_root,
        output_root=args.output_root,
        workers=max(1, args.workers),
        max_files=args.max_files,
        max_pairs=args.max_pairs,
        profile_samples=args.profile_samples,
    )
    print(
        f"measured {payload['pairs_measured']} of {payload['pairs_enumerated']} bounces "
        f"from {payload['files_read']} matches"
    )
    for surface, model in sorted(payload["surface_model"].items()):
        print(
            f"  {surface}: e_y {model['at_model_spin']['restitution_median']:.3f} "
            f"retention {model['at_model_spin']['retention_median']:.3f} n={model['n']}"
        )


if __name__ == "__main__":
    main()
