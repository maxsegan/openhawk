"""Regression-test the anchor-first fitter on closed-loop synthetic flights.

Twenty deterministic Arm-D flight windows supply their real boundaries, source PTS, and
per-frame camera matrices.  Trajectories use the repository drag/Magnus sampler and court
bounce law, are projected through those cameras, receive one-native-pixel Gaussian noise,
and are passed to the production fitter and flight ledger classifier.  The primary
checkerboard arm must accept at least 19/20 flights with a corpus-median held-out
median error no greater than two native pixels.
"""

from __future__ import annotations

import argparse
import copy
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq

from cv.pipeline import anchor_first_fit
from cv.pipeline.anchor_first_fit import _simulate
from cv.pipeline.flight_ledger import NET_CENTER_HEIGHT_M
from cv.pipeline.rich_ball_physics import ShotFit, project_one
from cv.validation.s6root_common import (
    BALL_RADIUS_M,
    NET_Y_M,
    ORACLE_RELATIVE,
    RESULT_RELATIVE,
    DoubledFrameCamera,
    FlightCase,
    clip_pts,
    compact_and_classify,
    data_root,
    frame_time,
    load_flight_cases,
    percentile,
    write_csv,
    write_json,
)

SEED = 20260902
TARGET_FLIGHTS = 20
PRIMARY_ARM = "production_checkerboard_nfev20_xscale_jac"
CONTACT_RAY_ARM = "contact_ray_checkerboard_nfev20_xscale_jac"
NET_CONSTRAINT_ARM = "net_constraint_soft_contact_nfev20_xscale_jac"
NET_CONSTRAINT_CONTACT_RAY_ARM = "net_constraint_hard_contact_ray_nfev20_xscale_jac"
MIN_ACCEPTED_FLIGHTS = 19
MAX_HELD_OUT_MEDIAN_PX = 2.0


def assert_regression_target(payload: dict[str, Any]) -> None:
    """Raise when the production checkerboard arm does not close synthetically."""
    primary = next((row for row in payload["table"] if row["arm"] == PRIMARY_ARM), None)
    if primary is None:
        raise AssertionError(f"closed-loop table is missing primary arm {PRIMARY_ARM!r}")
    accepted = int(primary["accepted"])
    held_out_median = primary["held_out_median_px_corpus_median"]
    if accepted < MIN_ACCEPTED_FLIGHTS or held_out_median is None:
        raise AssertionError(
            "anchor-first closed loop failed: "
            f"accepted {accepted}/{TARGET_FLIGHTS}, expected >= {MIN_ACCEPTED_FLIGHTS}; "
            f"held-out median {held_out_median!r} px, expected <= {MAX_HELD_OUT_MEDIAN_PX:.1f} px"
        )
    if float(held_out_median) > MAX_HELD_OUT_MEDIAN_PX:
        raise AssertionError(
            "anchor-first closed loop failed: "
            f"accepted {accepted}/{TARGET_FLIGHTS}; held-out median "
            f"{float(held_out_median):.3f} px, expected <= {MAX_HELD_OUT_MEDIAN_PX:.1f} px"
        )
    contact_ray = next((row for row in payload["table"] if row["arm"] == CONTACT_RAY_ARM), None)
    if contact_ray is None:
        raise AssertionError(f"closed-loop table is missing contact-ray arm {CONTACT_RAY_ARM!r}")
    if (
        int(contact_ray["accepted"]) < MIN_ACCEPTED_FLIGHTS
        or contact_ray["held_out_median_px_corpus_median"] is None
        or float(contact_ray["held_out_median_px_corpus_median"]) > MAX_HELD_OUT_MEDIAN_PX
    ):
        raise AssertionError(
            "contact-ray closed loop failed: "
            f"accepted {contact_ray['accepted']}/{TARGET_FLIGHTS}; held-out median "
            f"{contact_ray['held_out_median_px_corpus_median']!r} px"
        )
    for arm in (NET_CONSTRAINT_ARM, NET_CONSTRAINT_CONTACT_RAY_ARM):
        candidate = next((row for row in payload["table"] if row["arm"] == arm), None)
        if candidate is None:
            raise AssertionError(f"closed-loop table is missing net-constraint arm {arm!r}")
        if (
            int(candidate["accepted"]) < MIN_ACCEPTED_FLIGHTS
            or candidate["held_out_median_px_corpus_median"] is None
            or float(candidate["held_out_median_px_corpus_median"]) > MAX_HELD_OUT_MEDIAN_PX
        ):
            raise AssertionError(
                f"net-constraint closed loop failed for {arm}: "
                f"accepted {candidate['accepted']}/{TARGET_FLIGHTS}; held-out median "
                f"{candidate['held_out_median_px_corpus_median']!r} px"
            )


def _bounce_anchor(case: FlightCase) -> dict[str, Any] | None:
    rows = [row for row in case.anchors if row["type"] == "bounce"]
    return rows[0] if len(rows) == 1 else None


def _continuous_theta(case: FlightCase, theta: np.ndarray, bounce: dict[str, Any]) -> np.ndarray:
    """Adjust only launch vz so the free pre-bounce path reaches ball-radius height."""
    target_frame = float(bounce["frame"])

    def height(vz: float) -> float:
        candidate = theta.copy()
        candidate[5] = vz
        positions, *_ = _simulate(
            candidate,
            case.start_frame,
            np.asarray([target_frame]),
            case.fps,
            case.surface,
            None,
        )
        return float(positions[0, 2] - BALL_RADIUS_M)

    grid = np.linspace(-40.0, 40.0, 81)
    values = [height(float(value)) for value in grid]
    bracket = next(
        (
            (float(left), float(right))
            for left, right, f_left, f_right in zip(grid, grid[1:], values, values[1:])
            if f_left == 0.0 or f_left * f_right < 0.0
        ),
        None,
    )
    if bracket is None:
        raise ValueError("could not put synthetic pre-bounce path on the court plane")
    output = theta.copy()
    output[5] = brentq(height, *bracket)
    return output


def _net_anchor(
    case: FlightCase,
    theta: np.ndarray,
    bounce: dict[str, Any] | None,
) -> dict[str, Any]:
    dense = np.linspace(case.start_frame, case.end_frame, 1001)
    positions, *_ = _simulate(theta, case.start_frame, dense, case.fps, case.surface, bounce)
    delta = positions[:, 1] - NET_Y_M
    crossing = next(
        (
            index
            for index in range(len(dense) - 1)
            if delta[index] == 0.0 or delta[index] * delta[index + 1] < 0.0
        ),
        None,
    )
    if crossing is None:
        raise ValueError("synthetic path does not cross the net plane")
    fraction = abs(delta[crossing]) / max(abs(delta[crossing]) + abs(delta[crossing + 1]), 1e-12)
    frame = float(dense[crossing] * (1.0 - fraction) + dense[crossing + 1] * fraction)
    xyz = _simulate(
        theta,
        case.start_frame,
        np.asarray([frame]),
        case.fps,
        case.surface,
        bounce,
    )[0][0]
    if xyz[2] <= NET_CENTER_HEIGHT_M + BALL_RADIUS_M + 0.05:
        raise ValueError("synthetic path does not clear the net")
    return {
        "type": "net_crossing",
        "frame": frame,
        "true_crossing_frame": frame,
        "crossing_frame_bounds": [float(np.floor(frame)), float(np.ceil(frame))],
        "source_frames": [int(np.floor(frame)), int(np.ceil(frame))],
        "crossing_direction_y": (1 if positions[crossing + 1, 1] > positions[crossing, 1] else -1),
        "xyz": xyz.tolist(),
        "sigma_m": 0.01,
        "source": "closed_loop_truth",
    }


def synthesize(case: FlightCase, rng: np.random.Generator) -> dict[str, Any]:
    if case.prior_fit is None:
        raise ValueError("synthetic seed needs a solved Arm-D fit")
    theta = np.asarray(case.prior_fit["theta"], dtype=float)
    real_bounce = _bounce_anchor(case)
    bounce = None
    if real_bounce is not None:
        theta = _continuous_theta(case, theta, real_bounce)
        frame = float(real_bounce["frame"])
        xyz = _simulate(
            theta,
            case.start_frame,
            np.asarray([frame]),
            case.fps,
            case.surface,
            None,
        )[0][0]
        xyz[2] = BALL_RADIUS_M
        bounce = {
            "type": "bounce",
            "frame": frame,
            "xyz": xyz.tolist(),
            "x": xyz,
            "sigma_m": 0.01,
            "source": "closed_loop_truth",
        }
    net = _net_anchor(case, theta, bounce)
    frames = np.arange(np.ceil(case.start_frame), np.floor(case.end_frame) + 1, dtype=int)
    if len(frames) < 8:
        raise ValueError("synthetic window is too short")
    # PointCamera intentionally does not carry its source directory.  Recover it from the
    # oracle layout used to construct every case.
    match_dir = data_root() / ORACLE_RELATIVE / "truth_track_root" / case.match_id
    pts = clip_pts(match_dir, case.clip)
    query = np.asarray(
        [
            case.start_frame
            + (
                frame_time(pts, float(frame), case.fps)
                - frame_time(pts, case.start_frame, case.fps)
            )
            * case.fps
            for frame in frames
        ],
        dtype=float,
    )
    positions = _simulate(
        theta,
        case.start_frame,
        query,
        case.fps,
        case.surface,
        bounce,
    )[0]
    pixels = {
        int(frame): project_one(case.camera.p_at(float(frame)), xyz) + rng.normal(0.0, 1.0, 2)
        for frame, xyz in zip(frames, positions, strict=True)
    }
    if not all(np.isfinite(pixel).all() for pixel in pixels.values()):
        raise ValueError("non-finite synthetic projection")
    contacts = copy.deepcopy(case.contacts)
    contacts[0]["phase"] = "rally"
    contact_frames = np.asarray([float(row["frame"]) for row in contacts], dtype=float)
    contact_positions = _simulate(
        theta,
        case.start_frame,
        contact_frames,
        case.fps,
        case.surface,
        bounce,
    )[0]
    players: dict[str, dict[float, np.ndarray]] = {"near": {}, "far": {}}
    for contact, position in zip(contacts, contact_positions, strict=True):
        contact["event_frame"] = float(contact["frame"])
        if not contact.get("terminal"):
            contact["image_observation_override"] = (
                project_one(case.camera.p_at(float(contact["frame"])), position)
                + rng.normal(0.0, 1.0, 2)
            ).tolist()
            contact["image_observation_frame"] = float(contact["frame"])
            contact["image_observation_source_frames"] = []
            contact["image_observation_source"] = "closed_loop_contact_emission"
            contact["image_observation_confidence"] = 1.0
        side = str(contact.get("side"))
        if side in {"near", "far"}:
            players[side][float(contact["frame"])] = position[:2]
    anchors = [net] + (
        [{key: value for key, value in bounce.items() if key != "x"}] if bounce else []
    )
    return {
        "ball": pixels,
        "anchors": anchors,
        "contacts": contacts,
        "players": players,
        "source_pts": pts,
        "bounce": bounce is not None,
        "bounce_anchor": bounce,
        "theta": theta,
    }


def _true_initial_fit(
    case: FlightCase,
    synthetic: dict[str, Any],
    *,
    doubled: bool = False,
) -> ShotFit:
    frames = np.asarray(sorted(synthetic["ball"]), dtype=float)
    physics_frames = frames / 2.0 if doubled else frames
    bounce = synthetic["bounce_anchor"]
    positions, velocities, spins, bounces, _ = _simulate(
        synthetic["theta"],
        case.start_frame,
        physics_frames,
        case.fps,
        case.surface,
        bounce,
    )
    fit = ShotFit(
        0,
        1,
        synthetic["theta"].copy(),
        0.0,
        len(frames),
        positions,
        velocities,
        frames,
        bounces,
        0,
    )
    object.__setattr__(fit, "_f0", case.start_frame * (2.0 if doubled else 1.0))
    object.__setattr__(fit, "_anchor_first_free", bounce is None)
    if bounce is not None:
        mapped = copy.deepcopy(bounce)
        if doubled:
            mapped["frame"] *= 2.0
        object.__setattr__(fit, "_fixed_bounce_anchor", mapped)
    return fit


@contextmanager
def _force_x_scale(value: str | float):
    """Validation-only interception; production source and objective stay unchanged."""
    original = anchor_first_fit.least_squares

    def wrapped(*args: Any, **kwargs: Any):
        kwargs["x_scale"] = value
        return original(*args, **kwargs)

    anchor_first_fit.least_squares = wrapped
    try:
        yield
    finally:
        anchor_first_fit.least_squares = original


def _doubled_case(
    case: FlightCase,
    synthetic: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[FlightCase, dict[str, Any]]:
    """Put native frames on even ticks; new half-frames become held-out witnesses."""
    f0 = case.start_frame
    f1 = case.end_frame
    ticks = np.arange(np.ceil(2.0 * f0), np.floor(2.0 * f1) + 1, dtype=int)
    physical_frames = ticks.astype(float) / 2.0
    positions = _simulate(
        synthetic["theta"],
        f0,
        physical_frames,
        case.fps,
        case.surface,
        synthetic["bounce_anchor"],
    )[0]
    ball = {
        int(tick): project_one(case.camera.p_at(float(frame)), xyz) + rng.normal(0.0, 1.0, 2)
        for tick, frame, xyz in zip(ticks, physical_frames, positions, strict=True)
    }
    contacts = copy.deepcopy(synthetic["contacts"])
    for contact in contacts:
        contact["frame"] *= 2.0
    anchors = copy.deepcopy(synthetic["anchors"])
    for anchor in anchors:
        anchor["frame"] *= 2.0
        if anchor.get("true_crossing_frame") is not None:
            anchor["true_crossing_frame"] *= 2.0
        if anchor.get("crossing_frame_bounds") is not None:
            anchor["crossing_frame_bounds"] = [
                2.0 * float(value) for value in anchor["crossing_frame_bounds"]
            ]
        if anchor.get("source_frames") is not None and anchor.get("type") == "net_crossing":
            anchor["source_frames"] = [2 * int(value) for value in anchor["source_frames"]]
    mapped = copy.copy(case)
    mapped.ball = ball
    mapped.contacts = contacts
    mapped.anchors = anchors
    mapped.fps = case.fps * 2.0
    mapped.camera = DoubledFrameCamera(case.camera)
    mapped.players = {
        side: {2 * frame: xy for frame, xy in rows.items()} for side, rows in case.players.items()
    }
    mapped.observation_weights = {int(tick): 1.0 for tick in ticks}
    doubled_synthetic = dict(synthetic)
    doubled_synthetic.update({"ball": ball, "contacts": contacts, "anchors": anchors})
    if synthetic["bounce_anchor"] is not None:
        doubled_synthetic["bounce_anchor"] = copy.deepcopy(synthetic["bounce_anchor"])
        doubled_synthetic["bounce_anchor"]["frame"] *= 2.0
    return mapped, doubled_synthetic


VARIANTS = (
    {
        "arm": "production_checkerboard_nfev20_xscale_jac",
        "max_nfev": 20,
        "x_scale": "jac",
        "true_initial": False,
        "all_native_train": False,
        "contact_ray": False,
        "net_point_anchor": True,
    },
    {
        "arm": CONTACT_RAY_ARM,
        "max_nfev": 20,
        "x_scale": "jac",
        "true_initial": False,
        "all_native_train": False,
        "contact_ray": True,
        "net_point_anchor": True,
    },
    {
        "arm": NET_CONSTRAINT_ARM,
        "max_nfev": 20,
        "x_scale": "jac",
        "true_initial": False,
        "all_native_train": False,
        "contact_ray": False,
        "net_point_anchor": False,
    },
    {
        "arm": NET_CONSTRAINT_CONTACT_RAY_ARM,
        "max_nfev": 20,
        "x_scale": "jac",
        "true_initial": False,
        "all_native_train": False,
        "contact_ray": True,
        "net_point_anchor": False,
    },
    {
        "arm": "production_checkerboard_nfev100_xscale_jac",
        "max_nfev": 100,
        "x_scale": "jac",
        "true_initial": False,
        "all_native_train": False,
        "contact_ray": False,
        "net_point_anchor": True,
    },
    {
        "arm": "production_checkerboard_nfev100_xscale_1",
        "max_nfev": 100,
        "x_scale": 1.0,
        "true_initial": False,
        "all_native_train": False,
        "contact_ray": False,
        "net_point_anchor": True,
    },
    {
        "arm": "truth_initialized_checkerboard_nfev20_xscale_jac",
        "max_nfev": 20,
        "x_scale": "jac",
        "true_initial": True,
        "all_native_train": False,
        "contact_ray": False,
        "net_point_anchor": True,
    },
    {
        "arm": "all_native_train_half_frame_witness_nfev100",
        "max_nfev": 100,
        "x_scale": "jac",
        "true_initial": False,
        "all_native_train": True,
        "contact_ray": False,
        "net_point_anchor": True,
    },
)


def run(oracle_root: Path, output_root: Path) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    selected: list[tuple[FlightCase, dict[str, Any]]] = []
    rejected = []
    for case in load_flight_cases(oracle_root, solved_only=True):
        try:
            synthetic = synthesize(case, rng)
        except (ValueError, FloatingPointError) as exc:
            rejected.append({"flight_id": case.flight_id, "reason": str(exc)})
            continue
        selected.append((case, synthetic))
        if len(selected) == TARGET_FLIGHTS:
            break
    if len(selected) != TARGET_FLIGHTS:
        raise RuntimeError(f"only built {len(selected)} of {TARGET_FLIGHTS} synthetic flights")

    rows = []
    for variant_index, variant in enumerate(VARIANTS):
        for case_index, (case, synthetic) in enumerate(selected):
            diagnostic_case = copy.copy(case)
            diagnostic_case.contacts = synthetic["contacts"]
            diagnostic_case.anchors = synthetic["anchors"]
            diagnostic_case.ball = synthetic["ball"]
            diagnostic_case.players = synthetic["players"]
            diagnostic_case.observation_weights = {frame: 1.0 for frame in synthetic["ball"]}
            variant_synthetic = synthetic
            if variant["all_native_train"]:
                diagnostic_case, variant_synthetic = _doubled_case(
                    diagnostic_case,
                    synthetic,
                    np.random.default_rng(SEED + 1000 + case_index),
                )
            initial_fit = (
                _true_initial_fit(
                    case,
                    variant_synthetic,
                    doubled=bool(variant["all_native_train"]),
                )
                if variant["true_initial"]
                else None
            )
            with _force_x_scale(variant["x_scale"]):
                fit = anchor_first_fit.fit_anchor_first_shot(
                    0,
                    diagnostic_case.contacts,
                    diagnostic_case.ball,
                    diagnostic_case.players,
                    diagnostic_case.camera,
                    diagnostic_case.fps,
                    diagnostic_case.surface,
                    int(variant["max_nfev"]),
                    diagnostic_case.anchors,
                    observation_weights=diagnostic_case.observation_weights,
                    initial_fit=initial_fit,
                    net_point_anchor=bool(variant["net_point_anchor"]),
                )
                if variant["contact_ray"] and fit is not None:
                    fit = anchor_first_fit.refine_anchor_first_point({0: fit}).get(0)
            status, reasons, compact = compact_and_classify(diagnostic_case, fit)
            shared_contacts = list(getattr(fit, "_shared_contacts", [])) if fit is not None else []
            net_constraint = getattr(fit, "_net_constraint", None) if fit is not None else None
            contact_anchor_errors = (
                [
                    float(row["error_m"])
                    for row in getattr(fit, "_anchor_errors", [])
                    if str(row.get("type", "")).startswith("contact_ray")
                ]
                if fit is not None
                else []
            )
            summary = (
                {
                    "arm": variant["arm"],
                    "held_out_reprojection_median_px": None,
                    "held_out_reprojection_p90_px": None,
                    "anchor_max_error_m": None,
                    "anchor_satisfied": False,
                }
                if fit is None
                else {
                    key: value
                    for key, value in compact.items()
                    if key
                    in {
                        "held_out_reprojection_median_px",
                        "held_out_reprojection_p90_px",
                        "anchor_max_error_m",
                        "anchor_satisfied",
                    }
                }
            )
            pts = synthetic["source_pts"]
            nominal = np.arange(len(pts), dtype=float) / case.fps
            rows.append(
                {
                    "arm": variant["arm"],
                    "flight_id": case.flight_id,
                    "match_id": case.match_id,
                    "fps": case.fps,
                    "start_frame": case.start_frame,
                    "end_frame": case.end_frame,
                    "bounce": variant_synthetic["bounce"],
                    "observations": len(variant_synthetic["ball"]),
                    "source_pts_max_abs_delta_from_index_ms": float(
                        np.max(np.abs((pts - pts[0]) - nominal[: len(pts)])) * 1000.0
                    ),
                    "solved": fit is not None,
                    "accepted": status == "provisional_valid",
                    "status": status,
                    "reasons": ";".join(reasons),
                    "held_out_median_px": summary["held_out_reprojection_median_px"],
                    "held_out_p90_px": summary["held_out_reprojection_p90_px"],
                    "anchor_max_error_m": summary["anchor_max_error_m"],
                    "true_net_crossing_frame": next(
                        float(row["true_crossing_frame"])
                        for row in variant_synthetic["anchors"]
                        if row.get("type") == "net_crossing"
                    ),
                    "fitted_net_crossing_frame": (
                        float(net_constraint["frame"]) if net_constraint is not None else None
                    ),
                    "net_crossing_time_error_frames": (
                        abs(
                            float(net_constraint["frame"])
                            - next(
                                float(row["true_crossing_frame"])
                                for row in variant_synthetic["anchors"]
                                if row.get("type") == "net_crossing"
                            )
                        )
                        if net_constraint is not None
                        else None
                    ),
                    "net_constraint_satisfied": (
                        bool(net_constraint.get("satisfied"))
                        if net_constraint is not None
                        else None
                    ),
                    "contact_ray_anchors": len(shared_contacts),
                    "contact_ray_boundary_offsets_frames": [
                        float(row["time_offset_frames"]) for row in shared_contacts
                    ],
                    "contact_ray_max_anchor_error_m": max(contact_anchor_errors, default=None),
                    "optimizer_nfev": compact.get("optimizer_nfev") if compact else None,
                }
            )

    table = []
    for variant in VARIANTS:
        selected_rows = [row for row in rows if row["arm"] == variant["arm"]]
        medians = [
            row["held_out_median_px"]
            for row in selected_rows
            if row["held_out_median_px"] is not None
        ]
        p90s = [
            row["held_out_p90_px"] for row in selected_rows if row["held_out_p90_px"] is not None
        ]
        crossing_errors = [
            row["net_crossing_time_error_frames"]
            for row in selected_rows
            if row["net_crossing_time_error_frames"] is not None
        ]
        table.append(
            {
                "arm": variant["arm"],
                "flights": len(selected_rows),
                "solved": sum(row["solved"] for row in selected_rows),
                "accepted": sum(row["accepted"] for row in selected_rows),
                "held_out_median_px_corpus_median": percentile(medians, 50.0),
                "held_out_median_px_corpus_p90": percentile(medians, 90.0),
                "held_out_p90_px_corpus_median": percentile(p90s, 50.0),
                "held_out_p90_px_corpus_p90": percentile(p90s, 90.0),
                "net_crossing_time_error_frames_median": percentile(crossing_errors, 50.0),
                "net_crossing_time_error_frames_p90": percentile(crossing_errors, 90.0),
                "net_constraint_satisfied": sum(
                    row["net_constraint_satisfied"] is True for row in selected_rows
                ),
            }
        )
    payload = {
        "schema": "s6root_closed_loop_v1",
        "artifact_class": "oracle_diagnostic",
        "seed": SEED,
        "noise_sigma_px": 1.0,
        "selection": "first 20 lexicographic solved Arm-D flights that cross and clear the net",
        "physics": "physics.flight drag+Magnus through cv.pipeline.anchor_first_fit._simulate",
        "fitter": "cv.pipeline.anchor_first_fit.fit_anchor_first_shot",
        "classification": "cv.pipeline.flight_ledger.classify_flight",
        "regression_target": {
            "primary_arm": PRIMARY_ARM,
            "minimum_accepted_flights": MIN_ACCEPTED_FLIGHTS,
            "maximum_held_out_median_px": MAX_HELD_OUT_MEDIAN_PX,
            "contact_ray_arm": CONTACT_RAY_ARM,
            "net_constraint_arms": [NET_CONSTRAINT_ARM, NET_CONSTRAINT_CONTACT_RAY_ARM],
        },
        "table": table,
        "rows": rows,
        "selection_rejections": rejected,
    }
    write_json(output_root / "01_closed_loop.json", payload)
    write_csv(
        output_root / "01_closed_loop.csv",
        table,
        list(table[0]),
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = data_root()
    parser.add_argument("--oracle-root", type=Path, default=root / ORACLE_RELATIVE)
    parser.add_argument("--output-root", type=Path, default=root / RESULT_RELATIVE)
    parser.add_argument(
        "--allow-regression-failure",
        action="store_true",
        help="write a known-failing diagnostic baseline instead of enforcing the target",
    )
    args = parser.parse_args()
    payload = run(args.oracle_root, args.output_root)
    print(json.dumps(payload["table"], indent=2))
    if not args.allow_regression_failure:
        assert_regression_target(payload)


if __name__ == "__main__":
    main()
