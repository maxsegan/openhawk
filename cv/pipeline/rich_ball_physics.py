"""Bidirectional, per-frame-camera ball flight reconstruction for rich charts.

This is deliberately a new artifact rather than a rewrite of frozen B2 outputs.  B2
estimated each contact from one local, static-camera arc; at the weak end of a monocular
fit that lets a perfectly reprojection-compatible contact drift many metres up its camera
ray.  Here every live shot is fit through the moving per-frame projection and then fit a
second time after its endpoints have been reconciled with the adjacent shots.  Thus an
internal contact uses the complete incoming and outgoing paths plus a loose player-reach
prior.

Only alternating-side, live-play spans are fit.  Same-side fault/serve preparation and
long dead spans are emitted as explicit unsupported contacts, never as a fictitious ball
on the ground.  Outputs are versioned rich-chart inputs with per-frame 3D state, bounce
states, contact velocities, heuristic uncertainty, and the racket impulse implied by the
incoming/outgoing velocities.

Example diagnostic run (no Tier-1 labels are read):

  .venv/bin/python cv/pipeline/rich_ball_physics.py --match rg2025f --clips pt0092
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys

# Determinism: pin BLAS/OpenMP to a single thread BEFORE numpy/scipy load their backends.
# Multithreaded BLAS reduces dot-products in a nondeterministic order, so bit-for-bit results
# vary run to run; those sub-ULP differences cascade through least_squares and flip a fit
# across a hard acceptance threshold (the 20 m/s energy gate, the straightness cut), which is
# why identical contact CSVs previously emitted different contact sets. setdefault leaves an
# explicit caller override intact. We already parallelise across clips at the process level,
# so single-threaded linear algebra costs little throughput.
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "physics"))

import flight  # noqa: E402
import impact  # noqa: E402
import physics_events as pe  # noqa: E402
import resolution as res  # noqa: E402
from run_manifest import StageRun  # noqa: E402
from frame_identity import frame_number_from_name  # noqa: E402

# Shared S6 ball calibration; see physics.flight.default_params and impact.py.
R_BALL = 0.0325
SUBSTEPS = 4
# Candidate flight scope. Long lobs and drop-shot exchanges are resolved by the
# physical objective and downstream gate rather than rejected by duration alone.
MAX_LIVE_GAP_S = 3.5
MIN_OBS = 10
# An interior double-"unknown" span (both endpoints' side estimates abstained) is re-fit
# under both opposite-side hypotheses. Baseline refused all equal-label spans, which
# incidentally suppressed the server bouncing the ball pre-serve (genuine low-energy
# flights, 2-8 m/s). Re-admitting such a span therefore requires positive evidence the
# flight is a struck ball: its fitted launch speed must clear this bar. Trusted
# opposite-side spans keep the benefit of the doubt and are not energy-gated, so soft real
# shots there are unaffected.
AMBIG_SPEED_MIN_MS = 20.0
# A same-side fallback rescue (a mislabelled boundary serve) fits slow through wrong-side
# anchor geometry even when real; only reject parked-ball flicker there.
RESCUE_FALLBACK_SPEED_MIN_MS = 6.0
PLAYER_SIGMA_M = 0.9
CONTACT_ANCHOR_SIGMA_M = 0.03
PIXEL_SIGMA = 2.0
CONTACT_REPROJECTION_SIGMA_PX = 6.0
CONTACT_OBSERVATION_WINDOW_F = 3
MAX_SPEED_MS = 75.0
GROUND_TOLERANCE_M = 0.015
FIXED_BOUNCE_CONTINUITY_SIGMA_M = 0.01
MAX_FIXED_BOUNCE_CONTINUITY_M = 0.12
MIN_BOUNCE_DESCENT_MS = 0.05
MAX_BOUNCE_SPEED_RATIO = 1.02
MAX_CONTACT_REACH_M = 2.0
CONTACT_APPARENT_HEIGHT_TOLERANCE_M = 0.45
MAX_SIMULATED_BOUNCES = 8
BOUNCE_NODE_IMPACT_SLACK_SIGMA_MPS = 2.5
MAX_BOUNCE_NODE_IMPACT_SLACK_MPS = 7.5
MAX_BOUNCE_NODE_RMS_PX = 35.0
MAX_BOUNCE_NODE_RECOVERY_REACH_M = 2.5


class PhysicalFitError(ValueError):
    """A numerical fit violates a non-negotiable physical constraint."""


def contact_xy_bounds(player: np.ndarray, reach: float) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned optimizer bounds whose corners remain inside the reach circle."""
    axis_reach = reach / math.sqrt(2.0)
    player = np.asarray(player, float)
    return player - axis_reach, player + axis_reach


def contact_image_observation(
    ball: dict[int, np.ndarray],
    frame: float,
    maximum_delta: int = CONTACT_OBSERVATION_WINDOW_F,
) -> tuple[np.ndarray | None, tuple[int, ...]]:
    """Interpolate the smoothed 2D track at a possibly fractional event frame."""
    if not ball:
        return None, ()
    before = [candidate for candidate in ball if candidate <= frame]
    after = [candidate for candidate in ball if candidate >= frame]
    lower = max(before) if before else None
    upper = min(after) if after else None
    if (
        lower is not None
        and upper is not None
        and frame - lower <= maximum_delta
        and upper - frame <= maximum_delta
    ):
        if lower == upper:
            return np.asarray(ball[lower], float).copy(), (lower,)
        alpha = float((frame - lower) / (upper - lower))
        point = (1.0 - alpha) * np.asarray(ball[lower], float) + alpha * np.asarray(
            ball[upper], float
        )
        return point, (lower, upper)
    nearest_frame = min(ball, key=lambda candidate: abs(candidate - frame))
    if abs(nearest_frame - frame) > maximum_delta:
        return None, ()
    return np.asarray(ball[nearest_frame], float).copy(), (nearest_frame,)


def contact_image_residual(
    point: np.ndarray,
    contact: dict,
    ball: dict[int, np.ndarray],
    camera,
    observation_weights: dict[int, float] | None = None,
) -> tuple[np.ndarray, dict]:
    observation_frame = float(contact.get("image_observation_frame", contact["frame"]))
    override = contact.get("image_observation_override")
    if override is not None:
        observed = np.asarray(override, dtype=float)
        source_frames = tuple(contact.get("image_observation_source_frames", ()))
    else:
        observed, source_frames = contact_image_observation(ball, observation_frame)
    if observed is None:
        return np.zeros(2), {
            "available": False,
            "source_frames": [],
            "error_px": None,
        }
    projected = project_one(camera.p_at(observation_frame), np.asarray(point, float))
    weights = observation_weights or {}
    confidence = (
        float(np.clip(contact.get("image_observation_confidence", 1.0), 0.05, 1.0))
        if override is not None
        else float(
            np.mean([np.clip(weights.get(frame, 1.0), 0.05, 1.0) for frame in source_frames])
        )
    )
    delta = projected - observed
    return delta / CONTACT_REPROJECTION_SIGMA_PX * math.sqrt(confidence), {
        "available": True,
        "source_frames": list(source_frames),
        "observation_frame": observation_frame,
        "observed_uv": observed.tolist(),
        "projected_uv": projected.tolist(),
        "confidence": confidence,
        "error_px": float(np.linalg.norm(delta)),
    }


def contact_ray_candidates(
    contact: dict,
    seed: np.ndarray,
    ball: dict[int, np.ndarray],
    players: dict[str, dict[int, np.ndarray]],
    camera,
    observation_weights: dict[int, float] | None = None,
    pose_witness: dict | None = None,
) -> tuple[list[dict], dict]:
    """Enumerate player-reachable 3D points exactly on the event's observed camera ray."""
    if contact.get("terminal") or contact.get("side") not in {"near", "far"}:
        return [], {"available": False, "reason": "not_a_racket_contact"}
    observation_frame = float(contact.get("image_observation_frame", contact["frame"]))
    override = contact.get("image_observation_override")
    if override is not None:
        observed = np.asarray(override, dtype=float)
        source_frames = tuple(contact.get("image_observation_source_frames", ()))
    else:
        observed, source_frames = contact_image_observation(ball, observation_frame)
    player = nearest(players.get(contact["side"], {}), float(contact["frame"]))
    if observed is None or player is None:
        return [], {"available": False, "reason": "missing_ball_or_player"}
    weights = observation_weights or {}
    confidence = (
        float(np.clip(contact.get("image_observation_confidence", 1.0), 0.05, 1.0))
        if override is not None
        else float(
            np.mean([np.clip(weights.get(frame, 1.0), 0.05, 1.0) for frame in source_frames])
        )
    )
    side_confidence = float(contact.get("side_evidence", {}).get("confidence", 1.0))
    if confidence < 0.35 or side_confidence < 0.50:
        return [], {"available": False, "reason": "low_confidence_evidence"}
    reach = 1.6 if contact.get("phase") == "serve" else MAX_CONTACT_REACH_M
    minimum_z = 1.0 if contact.get("phase") == "serve" else 0.15
    target_z = 2.65 if contact.get("phase") == "serve" else 1.5
    wrist_xyz = (
        np.asarray(pose_witness["wrist_xyz"], dtype=float)
        if pose_witness is not None and pose_witness.get("wrist_xyz") is not None
        else None
    )
    racket_heads = [
        (
            np.asarray(row["head_center_xyz"], dtype=float),
            float(
                np.clip(
                    row.get("evidence_confidence", pose_witness.get("confidence", 0.0)), 0.0, 1.0
                )
            ),
        )
        for row in (pose_witness or {}).get("racket_head_hypotheses", [])
        if row.get("head_center_xyz") is not None
    ]
    pose_confidence = (
        float(np.clip(pose_witness.get("confidence", 0.0), 0.0, 1.0))
        if pose_witness is not None
        else 0.0
    )
    apparent_height = (
        float(pose_witness["apparent_ball_height_m"])
        if pose_witness is not None and pose_witness.get("apparent_ball_height_m") is not None
        else None
    )
    apparent_height_confidence = (
        float(np.clip(pose_witness.get("apparent_height_confidence", 0.0), 0.0, 1.0))
        if pose_witness is not None
        else 0.0
    )
    projection = camera.p_at(observation_frame)
    candidates = []
    for y in np.linspace(player[1] - reach, player[1] + reach, 161):
        point = ray_at_y(projection, observed, float(y))
        if point is None or not np.all(np.isfinite(point)):
            continue
        player_distance = float(np.linalg.norm(point[:2] - player))
        if player_distance > reach or not minimum_z <= point[2] <= 3.5:
            continue
        if (
            contact.get("phase") != "serve"
            and apparent_height is not None
            and apparent_height_confidence >= 0.45
            and abs(point[2] - apparent_height) > CONTACT_APPARENT_HEIGHT_TOLERANCE_M
        ):
            continue
        pose_distance = float(np.linalg.norm(point - wrist_xyz)) if wrist_xyz is not None else None
        racket_match = (
            min(
                (
                    float(np.linalg.norm(point - head)) / max(head_confidence, 0.10),
                    float(np.linalg.norm(point - head)),
                    head_confidence,
                )
                for head, head_confidence in racket_heads
            )
            if racket_heads
            else None
        )
        racket_distance = racket_match[1] if racket_match is not None else None
        racket_confidence = racket_match[2] if racket_match is not None else None
        pose_penalty = 0.0
        if pose_distance is not None:
            # The racket sweet spot is normally about one racket length from the
            # active wrist. Keep this soft because monocular wrist depth is noisy.
            pose_penalty = pose_confidence * abs(pose_distance - 0.68) / 0.22
            if not 0.20 <= pose_distance <= 1.20:
                pose_penalty += 2.5 * pose_confidence
        if racket_distance is not None:
            pose_penalty += racket_confidence * racket_distance / 0.24
            if racket_distance > 0.90:
                pose_penalty += 2.0 * racket_confidence
        toward_net = point[1] - player[1] if contact["side"] == "near" else player[1] - point[1]
        behind_body_penalty = max(0.0, -toward_net - 0.35) / 0.25
        apparent_height_penalty = (
            2.5
            * apparent_height_confidence
            * abs(point[2] - apparent_height)
            / CONTACT_APPARENT_HEIGHT_TOLERANCE_M
            if apparent_height is not None
            else 0.0
        )
        geometry_prior_score = float(
            np.linalg.norm((point - np.asarray(seed, float)) / np.array([0.5, 0.5, 0.75]))
            + 0.20 * abs(point[2] - target_z)
            + 0.10 * player_distance / reach
            + apparent_height_penalty
            + 0.35 * behind_body_penalty
        )
        score = geometry_prior_score + pose_penalty
        candidates.append(
            {
                "prior_score": score,
                "geometry_prior_score": geometry_prior_score,
                "point": point,
                "player_distance_m": player_distance,
                "wrist_distance_m": pose_distance,
                "racket_head_distance_m": racket_distance,
                "racket_head_evidence_confidence": racket_confidence,
                "toward_net_m": float(toward_net),
            }
        )
    if not candidates:
        return [], {"available": False, "reason": "no_reachable_ray_point"}
    candidates.sort(key=lambda row: row["prior_score"])
    return candidates, {
        "available": True,
        "reason": "exact_event_ray_candidates",
        "source_frames": list(source_frames),
        "observation_frame": observation_frame,
        "confidence": confidence,
        "side_confidence": side_confidence,
        "observed_uv": observed.tolist(),
        "pose_witness": pose_witness,
        "apparent_height_m": apparent_height,
        "apparent_height_confidence": apparent_height_confidence,
    }


def contact_ray_anchor(
    contact: dict,
    seed: np.ndarray,
    ball: dict[int, np.ndarray],
    players: dict[str, dict[int, np.ndarray]],
    camera,
    observation_weights: dict[int, float] | None = None,
    pose_witness: dict | None = None,
) -> tuple[np.ndarray | None, dict]:
    """Choose the strongest-prior point from the exact event-ray candidates."""
    candidates, diagnostic = contact_ray_candidates(
        contact,
        seed,
        ball,
        players,
        camera,
        observation_weights,
        pose_witness,
    )
    if not candidates:
        return None, diagnostic
    # A single adjacent flight cannot distinguish a physically plausible racket branch from a
    # monocular depth alias. Keep the geometry-only candidate authoritative here; racket evidence
    # remains available to the later two-flight junction search where both arcs can validate it.
    if (pose_witness or {}).get("racket_head_hypotheses"):
        body_safe = [row for row in candidates if row["toward_net_m"] >= -0.35]
        selected = min(body_safe or candidates, key=lambda row: row["geometry_prior_score"])
    else:
        selected = candidates[0]
    point = selected["point"]
    player_distance = selected["player_distance_m"]
    pose_distance = selected["wrist_distance_m"]
    observed = np.asarray(diagnostic["observed_uv"], dtype=float)
    projection = camera.p_at(float(contact.get("image_observation_frame", contact["frame"])))
    reprojection_error = float(np.linalg.norm(project_one(projection, point) - observed))
    return point, {
        **diagnostic,
        "reason": "exact_event_ray",
        "anchor_xyz": point.tolist(),
        "player_distance_m": player_distance,
        "toward_net_m": selected["toward_net_m"],
        "wrist_distance_m": pose_distance,
        "racket_head_distance_m": selected.get("racket_head_distance_m"),
        "racket_head_evidence_confidence": selected.get("racket_head_evidence_confidence"),
        "reprojection_error_px": reprojection_error,
    }


@dataclass(frozen=True)
class MatchSpec:
    out: str
    fps: float
    surface: str
    features: str
    ball: str
    players: str
    bounce_times: str
    output_dir: str
    frames_dir: str
    native_frames_dir: str


def match_spec(match: str) -> MatchSpec:
    processed = os.path.join(REPO, "data", "processed")
    if match == "rg2025f":
        return MatchSpec(
            out=os.path.join(processed, match),
            fps=50.0,
            surface="clay",
            features=os.path.join(processed, "bakeoff", "B2", "features_final.csv"),
            ball=os.path.join(processed, match, "ball_track_wasb_contact_v2_decoded.csv"),
            players=os.path.join(processed, match, "player_boxes_50_full_v2.csv"),
            bounce_times=os.path.join(processed, "bakeoff", "B2", "bounces_court.csv"),
            output_dir=os.path.join(processed, match),
            frames_dir=os.path.join(processed, match, "rally_frames_50_contact_v2"),
            native_frames_dir=os.path.join(processed, match, "rally_frames_50_1080"),
        )
    if match == "uso2025f":
        return MatchSpec(
            out=os.path.join(processed, match),
            fps=59.94005994005994,
            surface="hard",
            features=os.path.join(processed, "bakeoff", "B2", "uso", "features_uso_v1.csv"),
            ball=os.path.join(processed, match, "ball_track_wasb_full_v1.csv"),
            players=os.path.join(processed, match, "player_boxes_59.9401_full_v2.csv"),
            bounce_times=os.path.join(processed, "bakeoff", "B2", "uso", "bounces_uso_v1.csv"),
            output_dir=os.path.join(processed, match),
            frames_dir=os.path.join(processed, match, "rally_frames_60"),
            native_frames_dir=os.path.join(processed, match, "rally_frames_60_1080"),
        )
    raise ValueError(f"unsupported match: {match}")


def frame_num(raw: str) -> int:
    raw = raw.strip()
    return frame_number_from_name(raw) if raw.startswith("f_") else int(round(float(raw)))


def highest_frames(spec: MatchSpec):
    """Highest measured frame twin plus its explicit coordinate transform."""
    return res.select_highest_resolution_frame_dir((spec.native_frames_dir, spec.frames_dir))


def project_one(P: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    uvw = P @ np.r_[xyz, 1.0]
    return uvw[:2] / uvw[2]


def spin_vector(theta: np.ndarray) -> np.ndarray:
    return pe.spin_vector(theta)


def full_spin_theta(theta: np.ndarray) -> np.ndarray:
    """Upgrade legacy scalar-spin parameters to top/side/rifle components."""
    theta = np.asarray(theta, float)
    if len(theta) >= 9:
        return theta[:9].copy()
    if len(theta) != 7:
        raise ValueError(f"expected 7 or 9 trajectory parameters, got {len(theta)}")
    return np.r_[theta, 0.0, 0.0]


def spin_component_rpm(theta: np.ndarray) -> np.ndarray:
    upgraded = full_spin_theta(theta)
    return upgraded[6:9] * 100.0 * 60.0 / (2.0 * math.pi)


def signed_topspin(w: np.ndarray, v: np.ndarray) -> float:
    horizontal = math.hypot(v[0], v[1])
    if horizontal < 1e-6:
        return 0.0
    axis = np.array([-v[1] / horizontal, v[0] / horizontal, 0.0])
    return float(np.dot(w, axis))


def bounce_velocity(v: np.ndarray, w: np.ndarray, surface: str):
    if float(np.linalg.norm(v[:2])) < 0.1 or v[2] >= 0:
        return np.array([v[0], v[1], abs(v[2]) * 0.7]), w, "fallback"
    try:
        result = impact.court_bounce_vector(v, w, surface=surface)
    except ValueError:
        return np.array([v[0], v[1], abs(v[2]) * 0.7]), w, "fallback"
    return result.velocity, result.spin, result.regime


def bounce_energy_ratio(bounce: dict) -> float:
    """Rigid-body kinetic-energy ratio across a court impact."""

    def energy(velocity: np.ndarray, spin: np.ndarray) -> float:
        velocity = np.asarray(velocity, float)
        spin = np.asarray(spin, float)
        return 0.5 * float(velocity @ velocity) + 0.5 * impact.ALPHA * R_BALL**2 * float(
            spin @ spin
        )

    incoming = energy(bounce["v_in"], bounce.get("w_in", np.zeros(3)))
    outgoing = energy(bounce["v_out"], bounce.get("w_out", np.zeros(3)))
    return outgoing / max(incoming, 1e-9)


def _rk4(x: np.ndarray, v: np.ndarray, w: np.ndarray, dt: float):
    if (
        not np.all(np.isfinite(x))
        or not np.all(np.isfinite(v))
        or not np.all(np.isfinite(w))
        or float(np.linalg.norm(x)) > 1_000.0
        or float(np.linalg.norm(v)) > 250.0
        or float(np.linalg.norm(w)) > 5_000.0
    ):
        raise FloatingPointError("flight state diverged outside physical search bounds")
    lift = bool(np.linalg.norm(w) > 1.0)
    next_state = flight.rk4_step(x, v, w, dt, lift=lift)
    if not all(np.all(np.isfinite(row)) for row in next_state):
        raise FloatingPointError("flight integration produced a non-finite state")
    return next_state


def simulate_with_spin(
    theta: np.ndarray,
    f0: float,
    query_frames: np.ndarray,
    fps: float,
    surface: str,
):
    """Integrate position, velocity, and angular velocity through flight and bounces."""
    query = np.asarray(query_frames, float)
    if len(query) == 0 or float(query.min()) < f0 - 1e-6:
        raise ValueError("simulation queries must be at or after the contact")
    dt = 1.0 / (fps * SUBSTEPS)
    end_step = int(math.ceil((float(query.max()) - f0) * SUBSTEPS))
    xs = np.zeros((end_step + 1, 3))
    vs = np.zeros((end_step + 1, 3))
    ws = np.zeros((end_step + 1, 3))
    x = np.asarray(theta[:3], float).copy()
    v = np.asarray(theta[3:6], float).copy()
    w = spin_vector(theta)
    xs[0], vs[0], ws[0] = x, v, w
    bounces: list[dict] = []
    for step in range(end_step):
        xn, vn, wn = _rk4(x, v, w, dt)
        if xn[2] < R_BALL and vn[2] < 0 and len(bounces) < MAX_SIMULATED_BOUNCES:
            fraction = float(np.clip((x[2] - R_BALL) / max(x[2] - xn[2], 1e-9), 0, 1))
            xb = x + fraction * (xn - x)
            vb = v + fraction * (vn - v)
            wb = w + fraction * (wn - w)
            v_after, w_after, regime = bounce_velocity(vb, wb, surface)
            bounces.append(
                {
                    "frame": f0 + (step + fraction) / SUBSTEPS,
                    "x": xb.copy(),
                    "v_in": vb.copy(),
                    "v_out": v_after.copy(),
                    "w_in": wb.copy(),
                    "w_out": w_after.copy(),
                    "regime": regime,
                    "_step": step,
                    "_fraction": fraction,
                }
            )
            xn, vn, wn = _rk4(
                xb,
                v_after,
                w_after,
                dt * (1 - fraction),
            )
        elif xn[2] < R_BALL:
            xn[2] = R_BALL
            vn[2] = max(0.0, vn[2])
        x, v, w = xn, vn, wn
        xs[step + 1], vs[step + 1], ws[step + 1] = x, v, w
    q = (query - f0) * SUBSTEPS
    lo = np.floor(q).astype(int)
    hi = np.minimum(lo + 1, end_step)
    alpha = (q - lo)[:, None]
    sampled_x = (1 - alpha) * xs[lo] + alpha * xs[hi]
    sampled_v = (1 - alpha) * vs[lo] + alpha * vs[hi]
    sampled_w = (1 - alpha) * ws[lo] + alpha * ws[hi]
    events_by_step = {int(row["_step"]): row for row in bounces}
    for query_index, (step, fraction) in enumerate(zip(lo, alpha[:, 0])):
        bounce = events_by_step.get(int(step))
        if bounce is None:
            continue
        impact_fraction = float(bounce["_fraction"])
        if fraction <= impact_fraction and impact_fraction > 1e-12:
            local = fraction / impact_fraction
            sampled_x[query_index] = (1 - local) * xs[step] + local * bounce["x"]
            sampled_v[query_index] = (1 - local) * vs[step] + local * bounce["v_in"]
            sampled_w[query_index] = (1 - local) * ws[step] + local * bounce["w_in"]
        elif impact_fraction < 1.0 - 1e-12:
            local = (fraction - impact_fraction) / (1.0 - impact_fraction)
            sampled_x[query_index] = (1 - local) * bounce["x"] + local * xs[step + 1]
            sampled_v[query_index] = (1 - local) * bounce["v_out"] + local * vs[step + 1]
            sampled_w[query_index] = (1 - local) * bounce["w_out"] + local * ws[step + 1]
    for bounce in bounces:
        bounce.pop("_step", None)
        bounce.pop("_fraction", None)
    return sampled_x, sampled_v, sampled_w, bounces


def simulate(theta: np.ndarray, f0: float, query_frames: np.ndarray, fps: float, surface: str):
    """Backward-compatible position/velocity simulation."""
    x, v, _, bounces = simulate_with_spin(theta, f0, query_frames, fps, surface)
    return x, v, bounces


def simulate_free(theta: np.ndarray, f0: float, query_frames: np.ndarray, fps: float):
    """Bounceless RK4 flight state at frames before OR after f0 (positions only).

    A racket contact happens in mid-air, so within a few frames of a junction the fitted
    flight carries no bounce; integrating the pure drag+gravity+Magnus dynamics in whichever
    direction the query lies lets an incoming flight be extrapolated FORWARD past its end
    contact and an outgoing flight BACKWARD before its start contact. Used only for the short
    local extrapolation the closest-approach retimer needs — never for emission, which keeps
    the bounce-aware simulate(). Verified against simulate() to 0 error where no bounce falls
    in the window, and to round-trip exactly through f0."""
    query = np.asarray(query_frames, float)
    x0 = np.asarray(theta[:3], float).copy()
    v0 = np.asarray(theta[3:6], float).copy()
    w0 = spin_vector(theta)
    dt = 1.0 / (fps * SUBSTEPS)
    rel = (query - f0) * SUBSTEPS
    n_fwd = int(math.ceil(max(0.0, float(rel.max()))))
    n_bwd = int(math.ceil(max(0.0, float(-rel.min()))))
    xs_fwd = [x0.copy()]
    x, v, w = x0.copy(), v0.copy(), w0.copy()
    for _ in range(n_fwd):
        x, v, w = _rk4(x, v, w, dt)
        xs_fwd.append(x.copy())
    xs_bwd = []
    x, v, w = x0.copy(), v0.copy(), w0.copy()
    for _ in range(n_bwd):
        x, v, w = _rk4(x, v, w, -dt)
        xs_bwd.append(x.copy())
    grid = np.array(xs_bwd[::-1] + xs_fwd)  # grid[k] is substep (-n_bwd + k)
    qi = rel + n_bwd
    lo = np.clip(np.floor(qi).astype(int), 0, len(grid) - 1)
    hi = np.clip(lo + 1, 0, len(grid) - 1)
    alpha = np.clip(qi - np.floor(qi), 0.0, 1.0)[:, None]
    return (1 - alpha) * grid[lo] + alpha * grid[hi]


def _free_state_from(
    x0: np.ndarray,
    v0: np.ndarray,
    w: np.ndarray,
    query_steps: np.ndarray,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query_steps = np.asarray(query_steps, float)
    if len(query_steps) == 0:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3))
    if float(query_steps.min()) < -1e-9:
        raise ValueError("piecewise flight queries must be forward in time")
    end_step = int(math.ceil(float(query_steps.max()) * SUBSTEPS))
    dt = 1.0 / (fps * SUBSTEPS)
    positions = np.zeros((end_step + 1, 3))
    velocities = np.zeros((end_step + 1, 3))
    spins = np.zeros((end_step + 1, 3))
    x = np.asarray(x0, float).copy()
    v = np.asarray(v0, float).copy()
    positions[0], velocities[0], spins[0] = x, v, w
    for step in range(end_step):
        x, v, w = _rk4(x, v, w, dt)
        positions[step + 1], velocities[step + 1], spins[step + 1] = x, v, w
    query = query_steps * SUBSTEPS
    lo = np.floor(query).astype(int)
    hi = np.minimum(lo + 1, end_step)
    alpha = (query - lo)[:, None]
    return (
        (1 - alpha) * positions[lo] + alpha * positions[hi],
        (1 - alpha) * velocities[lo] + alpha * velocities[hi],
        (1 - alpha) * spins[lo] + alpha * spins[hi],
    )


def _free_state_bidirectional(
    x0: np.ndarray,
    v0: np.ndarray,
    w0: np.ndarray,
    query_steps: np.ndarray,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate a free-flight state to signed frame offsets from one exact node."""
    query_steps = np.asarray(query_steps, float)
    positions = np.empty((len(query_steps), 3))
    velocities = np.empty((len(query_steps), 3))
    spins = np.empty((len(query_steps), 3))
    for direction in (-1, 1):
        indices = np.flatnonzero(query_steps * direction >= -1e-9)
        if not len(indices):
            continue
        values = query_steps[indices] * direction
        order = np.argsort(values)
        ordered_indices = indices[order]
        ordered_values = values[order]
        end_step = int(math.ceil(float(ordered_values.max()) * SUBSTEPS))
        dt = direction / (fps * SUBSTEPS)
        states = [(np.asarray(x0, float), np.asarray(v0, float), np.asarray(w0, float))]
        x, v, w = (row.copy() for row in states[0])
        for _ in range(end_step):
            x, v, w = _rk4(x, v, w, dt)
            states.append((x.copy(), v.copy(), w.copy()))
        grid_x = np.asarray([row[0] for row in states])
        grid_v = np.asarray([row[1] for row in states])
        grid_w = np.asarray([row[2] for row in states])
        query = ordered_values * SUBSTEPS
        lower = np.floor(query).astype(int)
        upper = np.minimum(lower + 1, end_step)
        alpha = (query - lower)[:, None]
        positions[ordered_indices] = (1 - alpha) * grid_x[lower] + alpha * grid_x[upper]
        velocities[ordered_indices] = (1 - alpha) * grid_v[lower] + alpha * grid_v[upper]
        spins[ordered_indices] = (1 - alpha) * grid_w[lower] + alpha * grid_w[upper]
    return positions, velocities, spins


def simulate_fixed_bounce_with_spin(
    theta: np.ndarray,
    f0: float,
    query_frames: np.ndarray,
    fps: float,
    surface: str,
    bounce_anchor: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], np.ndarray]:
    """Integrate through an exact ground-plane bounce node.

    The pre-impact flight must meet the anchor through ``continuity`` residuals. The
    rendered/post-impact state starts exactly at the known ground point and applies the
    surface impact law, so a trusted bounce cannot drift to a different court location.
    """
    query = np.asarray(query_frames, float)
    bounce_frame = float(bounce_anchor["frame"])
    bounce_position = np.asarray(bounce_anchor["x"], float)
    if bounce_frame <= f0:
        raise ValueError("fixed bounce must follow the flight start")
    w_before = spin_vector(theta)
    pre_query = query[query < bounce_frame]
    pre_positions, pre_velocities, pre_spins = _free_state_from(
        theta[:3],
        theta[3:6],
        w_before,
        pre_query - f0,
        fps,
    )
    impact_position, impact_velocity, impact_spin = _free_state_from(
        theta[:3],
        theta[3:6],
        w_before,
        np.array([bounce_frame - f0]),
        fps,
    )
    velocity_after, w_after, regime = bounce_velocity(
        impact_velocity[0],
        impact_spin[0],
        surface,
    )
    post_query = query[query >= bounce_frame]
    post_positions, post_velocities, post_spins = _free_state_from(
        bounce_position,
        velocity_after,
        w_after,
        post_query - bounce_frame,
        fps,
    )
    positions = np.empty((len(query), 3))
    velocities = np.empty((len(query), 3))
    spins = np.empty((len(query), 3))
    positions[query < bounce_frame] = pre_positions
    velocities[query < bounce_frame] = pre_velocities
    spins[query < bounce_frame] = pre_spins
    positions[query >= bounce_frame] = post_positions
    velocities[query >= bounce_frame] = post_velocities
    spins[query >= bounce_frame] = post_spins
    bounce = {
        "frame": bounce_frame,
        "x": bounce_position.copy(),
        "v_in": impact_velocity[0].copy(),
        "v_out": velocity_after.copy(),
        "w_in": impact_spin[0].copy(),
        "w_out": w_after.copy(),
        "regime": regime,
    }
    return (
        positions,
        velocities,
        spins,
        [bounce],
        impact_position[0] - bounce_position,
    )


def simulate_hard_bounce_knot(
    incoming_velocity: np.ndarray,
    incoming_spin: np.ndarray,
    f0: float,
    query_frames: np.ndarray,
    fps: float,
    surface: str,
    bounce_anchor: dict,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Integrate both sides of an exact observed court-impact state.

    Position at the impact is eliminated from the optimizer: it is always the supplied
    ground-plane anchor.  Incoming velocity and spin define the pre-impact path by
    backward RK4 integration, while the existing court-impact law defines the outgoing
    state and forward path.  This makes bounce continuity structural rather than a
    residual that robust loss can trade against image observations.
    """
    query = np.asarray(query_frames, float)
    bounce_frame = float(bounce_anchor["frame"])
    bounce_position = np.asarray(bounce_anchor["x"], float)
    incoming_velocity = np.asarray(incoming_velocity, float)
    incoming_spin = np.asarray(incoming_spin, float)
    if bounce_frame <= f0:
        raise ValueError("hard bounce must follow the flight start")
    if not np.isclose(float(bounce_position[2]), R_BALL, atol=0.02):
        raise ValueError("hard bounce anchor must lie on the ball-center ground plane")

    outgoing_velocity, outgoing_spin, regime = bounce_velocity(
        incoming_velocity,
        incoming_spin,
        surface,
    )
    before = query < bounce_frame
    positions = np.empty((len(query), 3), dtype=float)
    velocities = np.empty_like(positions)
    spins = np.empty_like(positions)
    if np.any(before):
        positions[before], velocities[before], spins[before] = _free_state_bidirectional(
            bounce_position,
            incoming_velocity,
            incoming_spin,
            query[before] - bounce_frame,
            fps,
        )
    if np.any(~before):
        positions[~before], velocities[~before], spins[~before] = _free_state_bidirectional(
            bounce_position,
            outgoing_velocity,
            outgoing_spin,
            query[~before] - bounce_frame,
            fps,
        )
    initial_position, initial_velocity, initial_spin = _free_state_bidirectional(
        bounce_position,
        incoming_velocity,
        incoming_spin,
        np.asarray([f0 - bounce_frame]),
        fps,
    )
    bounce = {
        "frame": bounce_frame,
        "x": bounce_position.copy(),
        "v_in": incoming_velocity.copy(),
        "v_out": outgoing_velocity.copy(),
        "w_in": incoming_spin.copy(),
        "w_out": outgoing_spin.copy(),
        "regime": regime,
    }
    return (
        positions,
        velocities,
        spins,
        [bounce],
        (initial_position[0], initial_velocity[0], initial_spin[0]),
    )


def simulate_fixed_bounce(
    theta: np.ndarray,
    f0: float,
    query_frames: np.ndarray,
    fps: float,
    surface: str,
    bounce_anchor: dict,
) -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray]:
    """Backward-compatible exact-bounce simulation."""
    x, v, _, bounces, continuity = simulate_fixed_bounce_with_spin(
        theta,
        f0,
        query_frames,
        fps,
        surface,
        bounce_anchor,
    )
    return x, v, bounces, continuity


def fixed_bounce_diagnostics(
    positions: np.ndarray,
    bounce: dict,
    continuity: np.ndarray,
) -> dict[str, float | bool]:
    incoming_speed = float(np.linalg.norm(bounce["v_in"]))
    outgoing_speed = float(np.linalg.norm(bounce["v_out"]))
    continuity_m = float(np.linalg.norm(continuity))
    minimum_height_m = float(np.min(np.asarray(positions, float)[:, 2]))
    speed_ratio = outgoing_speed / max(incoming_speed, 1e-9)
    energy_ratio = bounce_energy_ratio(bounce)
    descending = float(bounce["v_in"][2]) <= -MIN_BOUNCE_DESCENT_MS
    valid = (
        minimum_height_m >= R_BALL - GROUND_TOLERANCE_M
        and continuity_m <= MAX_FIXED_BOUNCE_CONTINUITY_M
        and descending
        and energy_ratio <= MAX_BOUNCE_SPEED_RATIO
    )
    return {
        "valid": valid,
        "minimum_height_m": minimum_height_m,
        "continuity_m": continuity_m,
        "incoming_vertical_speed_ms": float(bounce["v_in"][2]),
        "speed_ratio": speed_ratio,
        "energy_ratio": energy_ratio,
    }


class FrameCamera:
    def __init__(self, out: str, clip: str, p_file: str = "camera_P_per_frame_v1.npz"):
        p_data = np.load(os.path.join(out, p_file))
        # H comes from the version-matched court solve so that a v2 projection uses v2
        # homographies (p_at and h_at must be self-consistent). Default v1 -> v1 unchanged.
        h_file = p_file.replace("camera_P_per_frame", "court_H_per_frame")
        h_data = np.load(os.path.join(out, h_file))
        p_sel, h_sel = p_data["clips"] == clip, h_data["clips"] == clip
        self.P = dict(zip(p_data["frames"][p_sel].astype(int), p_data["P"][p_sel]))
        self.H = dict(zip(h_data["frames"][h_sel].astype(int), h_data["H"][h_sel]))

    def available(self) -> bool:
        return bool(self.P) and bool(self.H)

    def p_at(self, frame: float) -> np.ndarray:
        f = int(round(frame))
        if f in self.P:
            return self.P[f]
        nearest = min(self.P, key=lambda q: abs(q - f))
        return self.P[nearest]

    def h_at(self, frame: float) -> np.ndarray:
        f = int(round(frame))
        if f in self.H:
            return self.H[f]
        nearest = min(self.H, key=lambda q: abs(q - f))
        return self.H[nearest]


def ray_at_y(P: np.ndarray, uv: np.ndarray, y: float) -> np.ndarray | None:
    u, v = uv
    A = np.array(
        [
            [P[0, 0] - u * P[2, 0], P[0, 2] - u * P[2, 2]],
            [P[1, 0] - v * P[2, 0], P[1, 2] - v * P[2, 2]],
        ]
    )
    b = -np.array(
        [
            P[0, 1] * y + P[0, 3] - u * (P[2, 1] * y + P[2, 3]),
            P[1, 1] * y + P[1, 3] - v * (P[2, 1] * y + P[2, 3]),
        ]
    )
    try:
        xz = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return np.array([xz[0], y, xz[1]])


def load_rows(path: str, clip: str) -> list[dict]:
    with open(path, newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["clip"] == clip]


def load_contacts(path: str, clip: str) -> list[dict]:
    contacts = []
    for row in load_rows(path, clip):
        if row.get("veto", "0") == "1":
            continue
        value = row.get("frame_refined_float") or row.get("frame_refined") or row["frame_A"]
        contacts.append(
            {
                "frame": float(value),
                "frame_detector": float(value),
                "side": row.get("side_phys") or "unknown",
                "phase": row.get("phase_A") or "rally",
                "source": row.get("source_A", ""),
                "row": row,
            }
        )
    return sorted(contacts, key=lambda row: row["frame"])


def retime_contacts(contacts: list[dict], ball: dict[int, np.ndarray], half_window: int = 5):
    """Detector-blind sub-frame junction timing from incoming/outgoing pixel paths.

    A local quadratic is fit on either side without using the central four frames, where
    WASB commonly switches identity at racket occlusion.  Their intersection retimes a
    contact only when the two extrapolations agree within six pixels.  Serves without an
    incoming live path and contacts bordering a broken alternation keep detector timing.
    """
    for i in range(1, len(contacts) - 1):
        contact = contacts[i]
        if contacts[i - 1]["side"] == contact["side"] or contacts[i + 1]["side"] == contact["side"]:
            continue
        center = contact["frame_detector"]
        before = [f for f in ball if center - 14 <= f <= center - 2]
        after = [f for f in ball if center + 2 <= f <= center + 14]
        if len(before) < 6 or len(after) < 6:
            continue
        pre = [
            np.polyfit(np.asarray(before) - center, [ball[f][axis] for f in before], 2)
            for axis in range(2)
        ]
        post = [
            np.polyfit(np.asarray(after) - center, [ball[f][axis] for f in after], 2)
            for axis in range(2)
        ]
        offsets = np.linspace(-half_window, half_window, 161)
        distances = []
        for offset in offsets:
            a = np.array([np.polyval(poly, offset) for poly in pre])
            b = np.array([np.polyval(poly, offset) for poly in post])
            distances.append(float(np.linalg.norm(a - b)))
        best = int(np.argmin(distances))
        if distances[best] <= 6.0:
            contact["frame"] = center + float(offsets[best])
            contact["retime_residual_px"] = distances[best]
            contact["retime_source"] = "bidirectional_pixel_intersection"
            plausible = offsets[np.asarray(distances) <= distances[best] + 1.5]
            half_width = max(1.0, min(2.0, float((plausible.max() - plausible.min()) / 2)))
            contact["timing_ci_frames"] = half_width


# Advisor Q4 upgrade: junction timing was HYBRID — the bidirectional pixel-quadratic
# intersection above fired only where 6+ observations existed within 14 frames on both sides,
# else discovery's gap-midpoint flowed through. retime_closest_approach implements the advisor's
# closest-approach-of-refined-flights estimator (post-fit, image space). MEASURED on the dev-9
# labels (2026-07-21): once made bounce-aware it reaches PARITY with the incumbent at +/-2 (36/62)
# and +/-5 (46/62) but is marginally worse at +/-1 (27->25) and median matched offset (0.56->0.85f)
# — the jointly-refined flights already share the contact by construction, so their closest
# approach mostly reproduces the existing junction. It is therefore an env-gated LEVER, OFF by
# default (the equal-or-better pixel-quadratic/gap-midpoint stays); set RBP_CLOSEST_APPROACH=1 to
# enable. The first, bounceLESS, attempt regressed hard (+/-2 36->23) by extrapolating the incoming
# flight across its mid-flight bounce — see flight_positions.
CLOSEST_APPROACH = os.environ.get("RBP_CLOSEST_APPROACH", "0") == "1"


def retime_closest_approach(
    contacts: list[dict],
    fits: dict,
    camera: "FrameCamera",
    fps: float,
    surface: str,
    half_window: float = 6.0,
    samples: int = 241,
):
    """Advisor Q4 junction timing: the contact is the time of closest approach of the REFINED
    incoming and outgoing fitted flights, minimised over continuous time in IMAGE space.

    Both fitted flights (theta per shot — full drag+Magnus arcs, not local quadratics) are
    projected through the per-frame cameras and the offset minimising their pixel separation
    is the emitted contact time; image space dodges the monocular depth wobble a 3D distance
    would inherit. The incoming flight is extrapolated forward past its end contact and the
    outgoing backward before its start (simulate_free), so the two curves overlap across the
    junction. Runs post-fit and OVERRIDES retime_contacts wherever BOTH flights exist; the
    pixel-quadratic retime_contacts stays as the fallback for one-flight (serve/rescue)
    contacts. The interval half-width is read off the image-distance curvature at the minimum."""
    if not CLOSEST_APPROACH:
        return

    def flight_positions(fit, offsets):
        # Forward of the flight's launch: bounce-AWARE simulate (an incoming rally flight
        # spans a mid-flight bounce, so a bounceless extrapolation across it is garbage — the
        # cause of the first, regressing, image-space attempt). Before the launch: bounceless
        # simulate_free (the short pre-contact mid-air stretch carries no bounce).
        offsets = np.asarray(offsets, float)
        fwd = offsets >= fit.f0 - 1e-9
        pos = np.empty((len(offsets), 3))
        if fwd.any():
            xs, _, _ = simulate(fit.theta, fit.f0, offsets[fwd], fps, surface)
            pos[fwd] = xs
        if (~fwd).any():
            pos[~fwd] = simulate_free(fit.theta, fit.f0, offsets[~fwd], fps)
        return pos

    for i in range(len(contacts)):
        inc, out = fits.get(i - 1), fits.get(i)
        if inc is None or out is None:
            continue
        center = contacts[i]["frame"]
        offsets = np.linspace(center - half_window, center + half_window, samples)
        x_inc = flight_positions(inc, offsets)
        x_out = flight_positions(out, offsets)
        dists = np.array(
            [
                float(
                    np.linalg.norm(
                        project_one(camera.p_at(t), x_inc[k])
                        - project_one(camera.p_at(t), x_out[k])
                    )
                )
                for k, t in enumerate(offsets)
            ]
        )
        best = int(np.argmin(dists))
        contacts[i]["frame"] = float(offsets[best])
        contacts[i]["retime_residual_px"] = float(dists[best])
        contacts[i]["retime_source"] = "closest_approach_image"
        # Interval from the curvature of the image-distance well: the band of offsets whose
        # separation is within one pixel-sigma of the minimum is the timing ambiguity.
        band = offsets[dists <= dists[best] + PIXEL_SIGMA]
        half = float((band.max() - band.min()) / 2) if band.size else 1.0
        contacts[i]["timing_ci_frames"] = max(0.5, min(3.0, half))


def load_audio_onsets(spec: MatchSpec, clip: str):
    """Audio 'hit' onsets (sorted [(frame, confidence)]) for the timing witness, or []."""
    path = os.path.join(spec.out, "contact_events_audio_full_v1.csv")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("clip") == clip and row.get("kind") == "hit":
                out.append((float(row["frame"]), float(row.get("confidence", 0.0))))
    return sorted(out)


def load_witness(spec: MatchSpec, match: str):
    """Per-broadcast audio-witness calibration (constant A/V offset + tolerance), or None.

    The mic-position refit is deliberately NOT a fitted parameter here: on RG's clean events
    the sound-travel delay (0.8-2.5 f across the court) sits below the audio-onset noise floor
    (~1.5 f), so a fitted effective mic is non-identifiable (bootstrap x-std ~6.5 m on an 11 m
    court) and a constant offset wins cross-validation. The witness is therefore a constant mux
    plus a calibrated tolerance; physics-vs-audio disagreement beyond it FLAGS the contact for
    timing abstention (a flag, never a gate on emission)."""
    path = os.path.join(spec.out, f"audio_witness_{match}_v1.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def audio_witness_check(frame: float, onsets, witness, window: float = 8.0):
    """Nearest audio onset to a contact and its physics-vs-audio disagreement.

    Returns (onset_frame|None, delta|None, abstain_bool): delta = onset - (frame + mux);
    abstain = |delta| > tolerance. A contact with no nearby onset is not judged (abstain False)."""
    if witness is None or not onsets:
        return None, None, False
    best = min(onsets, key=lambda fc: abs(fc[0] - frame))
    if abs(best[0] - frame) > window:
        return None, None, False
    delta = best[0] - (frame + witness.get("mux_frames", 0.0))
    abstain = abs(delta) > witness.get("witness_tol_frames", 3.0)
    return best[0], delta, abstain


def load_ball(path: str, clip: str) -> dict[int, np.ndarray]:
    return {
        frame_num(row["frame"]): np.array([float(row["x"]), float(row["y"])])
        for row in load_rows(path, clip)
    }


def load_bounce_anchors(path: str, clip: str, ball: dict[int, np.ndarray], camera: FrameCamera):
    """B2 timing plus an exact ground position from the local raw ball observation."""
    anchors = []
    for row in load_rows(path, clip):
        frame = float(row["frame"])
        candidates = [f for f in ball if abs(f - frame) <= 4]
        if not candidates:
            continue
        observed_frame = max(candidates, key=lambda f: ball[f][1])
        uv = np.float32([[ball[observed_frame]]])
        xy = cv2.perspectiveTransform(uv, camera.h_at(observed_frame))[0, 0]
        anchors.append(
            {
                "frame": frame,
                "observed_frame": observed_frame,
                "x": np.array([float(xy[0]), float(xy[1]), R_BALL]),
            }
        )
    return anchors


def load_contact_geometry(path: str | None, clip: str):
    if not path or not os.path.exists(path):
        return {}
    geometry = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["clip"] != clip or not row.get("x", "").strip():
                continue
            geometry[int(row["contact_index"])] = {
                "center": np.array([float(row[k]) for k in ("x", "y", "z")]),
                "sigma": np.array([float(row[k]) for k in ("sigma_x", "sigma_y", "sigma_z")]),
                "source": "anthropometric_contact_volume",
            }
    return geometry


def bounce_for_shot(anchors: list[dict], f0: float, f1: float):
    candidates = [
        anchor
        for anchor in anchors
        if (
            f0 + (0.5 if anchor.get("hard_geometry") else 5)
            <= anchor["frame"]
            <= f1 - (0.5 if anchor.get("hard_geometry") else 5)
        )
    ]
    if not candidates:
        return None
    hard = [anchor for anchor in candidates if anchor.get("hard_geometry")]
    if hard:
        candidates = hard
    midpoint = (f0 + f1) / 2
    return min(candidates, key=lambda anchor: abs(anchor["frame"] - midpoint))


def load_players(path: str, clip: str, camera: FrameCamera):
    players: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for row in load_rows(path, clip):
        frame = frame_num(row["frame"])
        pixel = np.float32([[[(float(row["x0"]) + float(row["x1"])) / 2, float(row["y1"])]]])
        xy = cv2.perspectiveTransform(pixel, camera.h_at(frame))[0, 0]
        players[row["side"]][frame] = np.asarray(xy, float)
    return players


def nearest(mapping: dict[int, np.ndarray], frame: float, window: int = 8):
    if not mapping:
        return None
    f = min(mapping, key=lambda q: abs(q - frame))
    return mapping[f] if abs(f - frame) <= window else None


def contact_prior(contact: dict, ball: dict[int, np.ndarray], players, camera: FrameCamera):
    frame, side = contact["frame"], contact["side"]
    player = nearest(players.get(side, {}), frame)
    uv = nearest(ball, frame)
    if contact.get("terminal") and uv is not None:
        ground = cv2.perspectiveTransform(
            np.float32([[uv]]),
            camera.h_at(frame),
        )[0, 0]
        if np.all(np.isfinite(ground)):
            point = np.array([ground[0], ground[1], R_BALL], dtype=float)
            return point, point[:2].copy(), uv
    z_default = 2.65 if contact["phase"] == "serve" else 1.1
    if player is None:
        player = np.array([5.485, -1.0 if side == "near" else 24.77])
    point = None if uv is None else ray_at_y(camera.p_at(frame), uv, float(player[1]))
    if point is None or not np.all(np.isfinite(point)) or not (0.15 <= point[2] <= 3.5):
        point = np.array([player[0], player[1], z_default])
    point[0] = np.clip(point[0], player[0] - 1.5, player[0] + 1.5)
    point[1] = np.clip(point[1], player[1] - 1.5, player[1] + 1.5)
    return point, player, uv


@dataclass
class ShotFit:
    start: int
    end: int
    theta: np.ndarray
    rms_px: float
    n_obs: int
    xs_obs: np.ndarray
    vs_obs: np.ndarray
    obs_frames: np.ndarray
    bounces: list[dict]
    pass_index: int

    def state(self, frame: float, fps: float, surface: str):
        if getattr(self, "_anchor_first_free", False):
            position, velocity, _ = flight.sample_states(
                self.theta[:3],
                self.theta[3:6],
                spin_vector(self.theta),
                np.array([(float(frame) - self.f0) / fps]),
            )
            return position[0], velocity[0]
        net_split = getattr(self, "_net_split", None)
        if net_split is not None:
            segment = net_split["pre"] if frame <= net_split["frame"] else net_split["post"]
            return segment.state(frame, fps, surface)
        bounce_node = getattr(self, "_bounce_node_split", None)
        if bounce_node is not None:
            if bounce_node.get("sampling_model") == "measured_bounce_v1":
                elapsed = (float(frame) - float(bounce_node["frame"])) / fps
                dwell = float(bounce_node["dwell_seconds"])
                if not math.isfinite(dwell) or dwell < 0:
                    raise ValueError("invalid measured bounce dwell")
                before = elapsed <= 0.0
                sampler = flight.sample_signed_states if before else flight.sample_states
                position, velocity, _ = sampler(
                    bounce_node["xyz"],
                    bounce_node["incoming_velocity"]
                    if before
                    else bounce_node["outgoing_velocity"],
                    bounce_node["incoming_spin"] if before else bounce_node["outgoing_spin"],
                    np.array([elapsed if before else max(elapsed - dwell, 0.0)]),
                )
                return position[0], velocity[0]
            before = frame < bounce_node["frame"]
            position, velocity, _ = _free_state_bidirectional(
                bounce_node["xyz"],
                (bounce_node["incoming_velocity"] if before else bounce_node["outgoing_velocity"]),
                bounce_node["incoming_spin"] if before else bounce_node["outgoing_spin"],
                np.array([frame - bounce_node["frame"]]),
                fps,
            )
            return position[0], velocity[0]
        ballistic_node = getattr(self, "_ballistic_bounce_node", None)
        if ballistic_node is not None:
            before = frame < ballistic_node["frame"]
            elapsed = (frame - ballistic_node["frame"]) / fps
            velocity0 = (
                ballistic_node["incoming_velocity"]
                if before
                else ballistic_node["outgoing_velocity"]
            )
            acceleration = (
                ballistic_node["incoming_acceleration"]
                if before
                else ballistic_node["outgoing_acceleration"]
            )
            position = ballistic_node["xyz"] + velocity0 * elapsed + 0.5 * acceleration * elapsed**2
            return position, velocity0 + acceleration * elapsed
        fixed_bounce = getattr(self, "_fixed_bounce_anchor", None)
        if fixed_bounce is not None:
            x, v, _, _ = simulate_fixed_bounce(
                self.theta,
                self.f0,
                np.array([frame]),
                fps,
                surface,
                fixed_bounce,
            )
            return x[0], v[0]
        x, v, _ = simulate(
            self.theta, self.obs_frames[0] * 0 + self.f0, np.array([frame]), fps, surface
        )
        return x[0], v[0]

    @property
    def f0(self):
        return float(self._f0)  # assigned after construction; keeps CSV-centric dataclass small


def initial_theta(p0: np.ndarray, p1: np.ndarray, duration_s: float, old_row: dict):
    velocity = (p1 - p0) / max(duration_s, 0.2)
    # A one-bounce tennis flight needs a downward initial component less often than an
    # unconstrained endpoint parabola suggests.  Existing B2 velocity is a useful start,
    # never a residual or target.
    try:
        old = np.array(
            [float(old_row["vx_out"]), float(old_row["vy_out"]), float(old_row["vz_out"])]
        )
        if np.linalg.norm(old) < MAX_SPEED_MS:
            velocity = old
    except (KeyError, TypeError, ValueError):
        velocity[2] = (p1[2] - p0[2] + 4.905 * duration_s**2) / max(duration_s, 0.2)
    velocity[:2] = np.clip(velocity[:2], -55, 55)
    velocity[2] = np.clip(velocity[2], -25, 25)
    return np.r_[p0, velocity, 0.0, 0.0, 0.0]


def fit_shot(
    index: int,
    contacts: list[dict],
    ball: dict[int, np.ndarray],
    players,
    camera: FrameCamera,
    fps: float,
    surface: str,
    max_nfev: int,
    endpoint_anchors: tuple[np.ndarray, np.ndarray] | None = None,
    fixed_start_anchor: np.ndarray | None = None,
    seed: np.ndarray | None = None,
    pass_index: int = 1,
    side_override: tuple[str, str] | None = None,
    end_index: int | None = None,
    multi_start: bool = True,
    observation_weights: dict[int, float] | None = None,
) -> ShotFit | None:
    # end_index defaults to the adjacent contact. A rescue fit may span a non-adjacent
    # endpoint, treating any intervening same-side / too-close contact as a mid-flight
    # insertion (a phantom split of one continuous flight) rather than a span boundary,
    # so an inserted contact never removes a real neighbour's only fittable flight.
    if end_index is None:
        end_index = index + 1
    c0, c1 = contacts[index], contacts[end_index]
    if c0.get("side") not in {"near", "far"}:
        return None
    if not c1.get("terminal") and c1.get("side") not in {"near", "far"}:
        return None
    f0, f1 = c0["frame"], c1["frame"]
    # A live rally flight crosses the net, so its endpoints are on opposite sides. When the
    # discovery-stage labels disagree (same side, or an "unknown"), that is a labelling
    # conflict, not a physical impossibility: the caller re-fits under both opposite-side
    # player-anchor hypotheses and keeps the better-scoring geometry (side_override). The gap
    # gate still refuses genuinely disconnected pairs (a hidden contact lives between them).
    if f1 - f0 > MAX_LIVE_GAP_S * fps:
        return None
    frames = np.array(sorted(f for f in ball if math.ceil(f0) <= f <= math.floor(f1)), int)
    if len(frames) < MIN_OBS:
        return None
    uv = np.stack([ball[f] for f in frames])
    weights = np.array(
        [
            float(np.clip((observation_weights or {}).get(int(frame), 1.0), 0.05, 1.0))
            for frame in frames
        ]
    )
    if side_override is not None:
        c0 = {**c0, "side": side_override[0]}
        c1 = {**c1, "side": side_override[1]}
    p0, player0, _ = contact_prior(c0, ball, players, camera)
    p1, player1, _ = contact_prior(c1, ball, players, camera)
    theta0 = (
        full_spin_theta(seed)
        if seed is not None
        else initial_theta(p0, p1, (f1 - f0) / fps, c0["row"])
    )
    P_frames = [camera.p_at(int(f)) for f in frames]
    integration_frames = np.unique(np.r_[frames, f1])

    fixed_start = endpoint_anchors[0] if endpoint_anchors is not None else fixed_start_anchor

    def decode(params):
        return np.r_[fixed_start, params] if fixed_start is not None else params

    def residual(params):
        theta = decode(params)
        integrated_xs, _, bounces = simulate(theta, f0, integration_frames, fps, surface)
        xs = integrated_xs[: len(frames)]
        pixel = np.array([project_one(P, x) for P, x in zip(P_frames, xs)])
        parts = [(((pixel - uv) / PIXEL_SIGMA) * np.sqrt(weights)[:, None]).ravel()]
        end_x = integrated_xs[-1:]
        if endpoint_anchors is None:
            parts.append((theta[:2] - player0) / PLAYER_SIGMA_M)
            parts.append(
                np.zeros(2) if c1.get("terminal") else (end_x[0, :2] - player1) / PLAYER_SIGMA_M
            )
        else:
            parts.append((end_x[0] - endpoint_anchors[1]) / CONTACT_ANCHOR_SIGMA_M)
        speed = np.linalg.norm(theta[3:6])
        # Residual length must remain invariant across optimizer calls; bounce validity
        # is expressed as a fixed scalar, not one residual per detected bounce.
        bounce_penalty = sum(
            max(0.0, -b["x"][0])
            + max(0.0, b["x"][0] - 10.97)
            + max(0.0, -b["x"][1])
            + max(0.0, b["x"][1] - 23.77)
            for b in bounces
        )
        parts.append(
            np.array(
                [
                    max(0.0, 0.15 - theta[2]) / 0.1,
                    max(0.0, theta[2] - 3.5) / 0.05,
                    max(0.0, 0.15 - end_x[0, 2]) / 0.1,
                    max(0.0, end_x[0, 2] - 3.5) / 0.05,
                    max(0.0, speed - MAX_SPEED_MS) / 2.0,
                    0.15 * abs(theta[6]),
                    0.15 * abs(theta[7]),
                    0.20 * abs(theta[8]),
                    bounce_penalty,
                ]
            )
        )
        return np.concatenate(parts)

    lower = np.array([-3, -8, R_BALL, -70, -70, -40, -5, -5, -3], float)
    upper = np.array([14, 32, 3.55, 70, 70, 40, 5, 5, 3], float)
    start_reach = 1.6 if c0.get("phase") == "serve" else MAX_CONTACT_REACH_M
    if fixed_start is None:
        reach_lower, reach_upper = contact_xy_bounds(player0, start_reach)
        lower[:2] = np.maximum(lower[:2], reach_lower)
        upper[:2] = np.minimum(upper[:2], reach_upper)
        if c0.get("phase") == "serve":
            lower[2] = max(lower[2], 1.0)
    elif np.linalg.norm(np.asarray(fixed_start[:2]) - player0) > start_reach:
        return None
    starts = [theta0]
    if seed is None and multi_start:
        start_hypotheses = [
            (8, 0, 0),
            (14, 1.5, 0),
            (4, -1.5, 0),
        ]
        if c0.get("phase") == "serve":
            start_hypotheses.extend(
                [
                    (10, 1.5, 2.0),
                    (8, 0.5, -2.0),
                ]
            )
        for vz, topspin, sidespin in start_hypotheses:
            alt = theta0.copy()
            alt[5] = vz
            alt[6] = topspin
            alt[7] = sidespin
            starts.append(alt)
    best = None
    for start in starts:
        start_params = start[3:] if fixed_start is not None else start
        fit_lower = lower[3:] if fixed_start is not None else lower
        fit_upper = upper[3:] if fixed_start is not None else upper
        try:
            result = least_squares(
                residual,
                np.clip(start_params, fit_lower + 1e-5, fit_upper - 1e-5),
                bounds=(fit_lower, fit_upper),
                loss="soft_l1",
                f_scale=2.0,
                x_scale="jac",
                max_nfev=max_nfev,
            )
        except (ValueError, FloatingPointError):
            continue
        result_theta = decode(result.x)
        xs, vs, ws, bounces = simulate_with_spin(
            result_theta,
            f0,
            frames,
            fps,
            surface,
        )
        pixel = np.array([project_one(P, x) for P, x in zip(P_frames, xs)])
        squared = np.sum((pixel - uv) ** 2, axis=1)
        rms = float(np.sqrt(np.mean(squared)))
        weighted_rms = float(np.sqrt(np.average(squared, weights=weights)))
        end_x, _, _ = simulate(result_theta, f0, np.array([f1]), fps, surface)
        if not c1.get("terminal") and np.linalg.norm(end_x[0, :2] - player1) > MAX_CONTACT_REACH_M:
            continue
        score = weighted_rms + 0.1 * float(np.linalg.norm(result.fun[2 * len(frames) :]))
        if best is None or score < best[0]:
            best = score, result_theta, rms, weighted_rms, xs, vs, ws, bounces
    if best is None:
        return None
    _, theta, rms, weighted_rms, xs, vs, ws, bounces = best
    fit = ShotFit(index, end_index, theta, rms, len(frames), xs, vs, frames, bounces, pass_index)
    object.__setattr__(fit, "_f0", f0)
    object.__setattr__(fit, "_weighted_rms_px", weighted_rms)
    # best[0] folds the player-anchor residual into the pixel RMS, so a wrong-side
    # hypothesis that only fits pixels by ghosting depth scores worse than the true side.
    object.__setattr__(fit, "_score", float(best[0]))
    object.__setattr__(fit, "_ws_obs", ws)
    return fit


def fit_net_split_shot(
    index: int,
    contacts: list[dict],
    ball: dict[int, np.ndarray],
    camera: FrameCamera,
    fps: float,
    surface: str,
    seed: ShotFit,
    net_frame: float,
    net_xyz: np.ndarray,
    max_nfev: int,
    bounce_anchor: dict | None = None,
) -> ShotFit | None:
    """Fit independent pre/post-impact states joined at one observed net-plane point."""
    start_frame = float(contacts[index]["frame"])
    end_frame = float(contacts[index + 1]["frame"])
    if not start_frame < net_frame < end_frame:
        return None
    start_xyz = np.asarray(seed.theta[:3], float)
    end_xyz = np.asarray(seed.state(end_frame, fps, surface)[0], float)
    pre_contacts = [
        {**contacts[index], "frame": start_frame},
        {"frame": net_frame, "side": "net", "phase": "net_hit"},
    ]
    post_contacts = [
        {"frame": net_frame, "side": "net", "phase": "net_hit"},
        {**contacts[index + 1], "frame": end_frame},
    ]
    pre = shoot_shot(
        0,
        pre_contacts,
        ball,
        camera,
        fps,
        surface,
        (start_xyz, net_xyz),
        seed,
        max_nfev,
    )
    if pre is None or pre.bounces:
        return None
    post = shoot_shot(
        0,
        post_contacts,
        ball,
        camera,
        fps,
        surface,
        (net_xyz, end_xyz),
        seed,
        max_nfev,
        bounce_anchor=bounce_anchor,
    )
    if post is None and contacts[index + 1].get("terminal"):
        post = fit_fixed_start_terminal_segment(
            net_frame,
            end_frame,
            net_xyz,
            ball,
            camera,
            fps,
            surface,
            seed,
            max_nfev,
            bounce_anchor,
        )
    if post is None:
        return None
    frames = np.array(sorted(set(pre.obs_frames.tolist() + post.obs_frames.tolist())), dtype=int)
    positions = []
    velocities = []
    spins = []
    for frame in frames:
        segment = pre if frame <= net_frame else post
        position, velocity = segment.state(float(frame), fps, surface)
        positions.append(position)
        velocities.append(velocity)
        segment_spin = getattr(
            segment,
            "_ws_obs",
            np.repeat(spin_vector(segment.theta)[None, :], len(segment.obs_frames), axis=0),
        )
        nearest_index = int(np.argmin(np.abs(segment.obs_frames - frame)))
        spins.append(segment_spin[nearest_index])
    combined = ShotFit(
        index,
        index + 1,
        pre.theta.copy(),
        float(np.sqrt((pre.rms_px**2 + post.rms_px**2) / 2.0)),
        len(frames),
        np.asarray(positions),
        np.asarray(velocities),
        frames,
        post.bounces,
        2,
    )
    object.__setattr__(combined, "_f0", start_frame)
    object.__setattr__(combined, "_weighted_rms_px", combined.rms_px)
    object.__setattr__(combined, "_ws_obs", np.asarray(spins))
    object.__setattr__(
        combined,
        "_net_split",
        {
            "frame": float(net_frame),
            "xyz": np.asarray(net_xyz, float),
            "pre": pre,
            "post": post,
        },
    )
    post_diagnostics = getattr(post, "_physical_diagnostics", {})
    object.__setattr__(
        combined,
        "_physical_diagnostics",
        {
            "minimum_height_m": float(np.min(combined.xs_obs[:, 2])),
            "continuity_m": post_diagnostics.get("continuity_m"),
            "incoming_vertical_speed_ms": post_diagnostics.get("incoming_vertical_speed_ms"),
            "speed_ratio": post_diagnostics.get("speed_ratio"),
        },
    )
    fixed_bounce = getattr(post, "_fixed_bounce_anchor", None)
    if fixed_bounce is not None:
        object.__setattr__(combined, "_fixed_bounce_anchor", fixed_bounce)
    return combined


def fit_fixed_start_terminal_segment(
    start_frame: float,
    end_frame: float,
    start_xyz: np.ndarray,
    ball: dict[int, np.ndarray],
    camera: FrameCamera,
    fps: float,
    surface: str,
    seed: ShotFit,
    max_nfev: int,
    bounce_anchor: dict | None,
) -> ShotFit | None:
    """Fit a terminal segment from one exact impact node without inventing an endpoint."""
    frames = np.array(
        sorted(frame for frame in ball if math.ceil(start_frame) <= frame <= math.floor(end_frame)),
        dtype=int,
    )
    if len(frames) < MIN_OBS:
        return None
    observed = np.stack([ball[int(frame)] for frame in frames])
    cameras = [camera.p_at(frame) for frame in frames]
    seed_theta = full_spin_theta(seed.theta)
    starts = [
        seed_theta[3:9],
        np.r_[seed_theta[3:6], np.zeros(3)],
        np.array([0.0, 8.0, -1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, -8.0, -1.0, 0.0, 0.0, 0.0]),
    ]
    best = None
    for initial in starts:

        def residual(state):
            theta = np.r_[start_xyz, state]
            if bounce_anchor is not None:
                xs, _, _, _, continuity = simulate_fixed_bounce_with_spin(
                    theta,
                    start_frame,
                    frames,
                    fps,
                    surface,
                    bounce_anchor,
                )
            else:
                xs, _, _, _ = simulate_with_spin(
                    theta,
                    start_frame,
                    frames,
                    fps,
                    surface,
                )
                continuity = np.zeros(3)
            pixels = np.array([project_one(P, x) for P, x in zip(cameras, xs)])
            return np.r_[
                ((pixels - observed) / PIXEL_SIGMA).ravel(),
                continuity / FIXED_BOUNCE_CONTINUITY_SIGMA_M,
            ]

        try:
            result = least_squares(
                residual,
                np.clip(initial, [-70, -70, -40, -5, -5, -3], [70, 70, 40, 5, 5, 3]),
                bounds=([-70, -70, -40, -5, -5, -3], [70, 70, 40, 5, 5, 3]),
                loss="soft_l1",
                f_scale=2.0,
                x_scale="jac",
                max_nfev=max_nfev,
            )
        except (ValueError, FloatingPointError):
            continue
        theta = np.r_[start_xyz, result.x]
        if bounce_anchor is not None:
            xs, vs, ws, bounces, continuity = simulate_fixed_bounce_with_spin(
                theta,
                start_frame,
                frames,
                fps,
                surface,
                bounce_anchor,
            )
        else:
            xs, vs, ws, bounces = simulate_with_spin(
                theta,
                start_frame,
                frames,
                fps,
                surface,
            )
            continuity = np.zeros(3)
        pixels = np.array([project_one(P, x) for P, x in zip(cameras, xs)])
        rms = float(np.sqrt(np.mean(np.sum((pixels - observed) ** 2, axis=1))))
        score = rms + 100.0 * float(np.linalg.norm(continuity))
        if best is None or score < best[0]:
            best = score, theta, rms, xs, vs, ws, bounces, continuity
    if best is None:
        return None
    _, theta, rms, xs, vs, ws, bounces, continuity = best
    fit = ShotFit(0, 1, theta, rms, len(frames), xs, vs, frames, bounces, 2)
    object.__setattr__(fit, "_f0", start_frame)
    object.__setattr__(fit, "_weighted_rms_px", rms)
    object.__setattr__(fit, "_ws_obs", ws)
    object.__setattr__(
        fit,
        "_physical_diagnostics",
        {
            "minimum_height_m": float(np.min(xs[:, 2])),
            "continuity_m": float(np.linalg.norm(continuity)),
            "incoming_vertical_speed_ms": None,
            "speed_ratio": None,
        },
    )
    if bounce_anchor is not None:
        object.__setattr__(fit, "_fixed_bounce_anchor", bounce_anchor)
    return fit


def fit_bounce_node_shot(
    index: int,
    contacts: list[dict],
    ball: dict[int, np.ndarray],
    players: dict,
    camera: FrameCamera,
    fps: float,
    surface: str,
    bounce_anchor: dict,
    seed: ShotFit,
    max_nfev: int,
) -> ShotFit | None:
    """Fit both sides outward from an exact bounce with bounded impact-law slack."""
    start_frame = float(contacts[index]["frame"])
    end_frame = float(contacts[index + 1]["frame"])
    bounce_frame = float(bounce_anchor["frame"])
    if not start_frame < bounce_frame < end_frame:
        return None
    frames = np.array(
        sorted(frame for frame in ball if math.ceil(start_frame) <= frame <= math.floor(end_frame)),
        dtype=int,
    )
    if len(frames) < MIN_OBS:
        return None
    observed = np.stack([ball[int(frame)] for frame in frames])
    projections = [camera.p_at(frame) for frame in frames]
    node = np.asarray(bounce_anchor["x"], float)
    seed_theta = full_spin_theta(seed.theta)
    try:
        _, seed_velocity, seed_spin = _free_state_from(
            seed_theta[:3],
            seed_theta[3:6],
            spin_vector(seed_theta),
            np.array([bounce_frame - start_frame]),
            fps,
        )
        seed_in = seed_velocity[0]
    except (ValueError, FloatingPointError):
        seed_in = np.array([0.0, 20.0, -5.0])
    starts = [
        np.r_[seed_in, seed_theta[6:9], np.zeros(3)],
        np.r_[[0.0, -25.0, -5.0], np.zeros(3), np.zeros(3)],
        np.r_[[0.0, 25.0, -5.0], np.zeros(3), np.zeros(3)],
    ]
    start_player = nearest(players.get(contacts[index]["side"], {}), start_frame)
    end_player = nearest(players.get(contacts[index + 1]["side"], {}), end_frame)

    def states(parameters: np.ndarray):
        incoming_velocity = parameters[:3]
        spin_components = parameters[3:6]
        incoming_theta = np.r_[node, incoming_velocity, spin_components]
        incoming_spin = spin_vector(incoming_theta)
        expected_out, outgoing_spin, regime = bounce_velocity(
            incoming_velocity,
            incoming_spin,
            surface,
        )
        outgoing_velocity = expected_out + parameters[6:9]
        positions = np.empty((len(frames), 3))
        velocities = np.empty((len(frames), 3))
        spins = np.empty((len(frames), 3))
        before = frames < bounce_frame
        if np.any(before):
            positions[before], velocities[before], spins[before] = _free_state_bidirectional(
                node,
                incoming_velocity,
                incoming_spin,
                frames[before] - bounce_frame,
                fps,
            )
        if np.any(~before):
            positions[~before], velocities[~before], spins[~before] = _free_state_bidirectional(
                node,
                outgoing_velocity,
                outgoing_spin,
                frames[~before] - bounce_frame,
                fps,
            )
        return (
            positions,
            velocities,
            spins,
            incoming_spin,
            outgoing_velocity,
            outgoing_spin,
            regime,
        )

    def residual(parameters: np.ndarray) -> np.ndarray:
        positions, _, _, _, outgoing_velocity, _, _ = states(parameters)
        pixels = np.asarray(
            [project_one(projection, xyz) for projection, xyz in zip(projections, positions)]
        )
        pixels = np.nan_to_num(pixels, nan=1e4, posinf=1e4, neginf=-1e4)
        values = [((pixels - observed) / PIXEL_SIGMA).ravel()]
        values.append(parameters[6:9] / BOUNCE_NODE_IMPACT_SLACK_SIGMA_MPS)
        values.append(np.maximum(0.0, R_BALL - positions[:, 2]) / GROUND_TOLERANCE_M)
        values.append(
            np.array(
                [
                    max(0.0, parameters[2] + MIN_BOUNCE_DESCENT_MS),
                    max(0.0, np.linalg.norm(parameters[:3]) - MAX_SPEED_MS) / 2.0,
                    max(0.0, np.linalg.norm(outgoing_velocity) - MAX_SPEED_MS) / 2.0,
                ]
            )
        )
        if start_player is not None:
            values.append(
                np.array(
                    [
                        max(
                            0.0,
                            np.linalg.norm(positions[0, :2] - start_player) - MAX_CONTACT_REACH_M,
                        )
                        / 0.20
                    ]
                )
            )
        if end_player is not None and not contacts[index + 1].get("terminal"):
            values.append(
                np.array(
                    [
                        max(
                            0.0,
                            np.linalg.norm(positions[-1, :2] - end_player) - MAX_CONTACT_REACH_M,
                        )
                        / 0.20
                    ]
                )
            )
        return np.concatenate(values)

    best = None
    bounds = (
        np.array([-70.0, -70.0, -40.0, -5.0, -5.0, -3.0, -10.0, -10.0, -10.0]),
        np.array([70.0, 70.0, -MIN_BOUNCE_DESCENT_MS, 5.0, 5.0, 3.0, 10.0, 10.0, 10.0]),
    )
    for initial in starts:
        try:
            result = least_squares(
                residual,
                np.clip(initial, bounds[0] + 1e-5, bounds[1] - 1e-5),
                bounds=bounds,
                loss="soft_l1",
                f_scale=2.0,
                x_scale="jac",
                max_nfev=max(40, max_nfev),
            )
        except (ValueError, FloatingPointError):
            continue
        positions, velocities, spins, incoming_spin, outgoing_velocity, outgoing_spin, regime = (
            states(result.x)
        )
        pixels = np.asarray(
            [project_one(projection, xyz) for projection, xyz in zip(projections, positions)]
        )
        pixels = np.nan_to_num(pixels, nan=1e4, posinf=1e4, neginf=-1e4)
        rms = float(np.sqrt(np.mean(np.sum((pixels - observed) ** 2, axis=1))))
        impact_slack = float(np.linalg.norm(result.x[6:9]))
        score = rms + impact_slack / BOUNCE_NODE_IMPACT_SLACK_SIGMA_MPS
        if best is None or score < best[0]:
            best = (
                score,
                result.x,
                rms,
                positions,
                velocities,
                spins,
                incoming_spin,
                outgoing_velocity,
                outgoing_spin,
                regime,
            )
    if best is None:
        return None
    (
        _,
        parameters,
        rms,
        positions,
        velocities,
        spins,
        incoming_spin,
        outgoing_velocity,
        outgoing_spin,
        regime,
    ) = best
    impact_slack = float(np.linalg.norm(parameters[6:9]))
    minimum_height = float(np.min(positions[:, 2]))
    start_distance = (
        float(np.linalg.norm(positions[0, :2] - start_player)) if start_player is not None else 0.0
    )
    end_distance = (
        float(np.linalg.norm(positions[-1, :2] - end_player))
        if end_player is not None and not contacts[index + 1].get("terminal")
        else 0.0
    )
    screen_reasons = []
    if rms > MAX_BOUNCE_NODE_RMS_PX:
        screen_reasons.append("high_reprojection")
    if impact_slack > MAX_BOUNCE_NODE_IMPACT_SLACK_MPS:
        screen_reasons.append("impact_slack")
    if minimum_height < R_BALL - GROUND_TOLERANCE_M:
        screen_reasons.append("below_court")
    if start_distance > MAX_BOUNCE_NODE_RECOVERY_REACH_M:
        screen_reasons.append("start_outside_recovery_reach")
    if end_distance > MAX_BOUNCE_NODE_RECOVERY_REACH_M:
        screen_reasons.append("end_outside_recovery_reach")
    start_position, start_velocity, _ = _free_state_bidirectional(
        node,
        parameters[:3],
        incoming_spin,
        np.array([start_frame - bounce_frame]),
        fps,
    )
    start_theta = np.r_[start_position[0], start_velocity[0], parameters[3:6]]
    bounce = {
        "frame": bounce_frame,
        "x": node.copy(),
        "v_in": parameters[:3].copy(),
        "v_out": outgoing_velocity.copy(),
        "w_in": incoming_spin.copy(),
        "w_out": outgoing_spin.copy(),
        "regime": regime,
    }
    fit = ShotFit(
        index,
        index + 1,
        start_theta,
        rms,
        len(frames),
        positions,
        velocities,
        frames,
        [bounce],
        2,
    )
    object.__setattr__(fit, "_f0", start_frame)
    object.__setattr__(fit, "_weighted_rms_px", rms)
    object.__setattr__(fit, "_ws_obs", spins)
    object.__setattr__(fit, "_fixed_bounce_anchor", bounce_anchor)
    object.__setattr__(
        fit,
        "_bounce_node_split",
        {
            "frame": bounce_frame,
            "xyz": node.copy(),
            "incoming_velocity": parameters[:3].copy(),
            "incoming_spin": incoming_spin.copy(),
            "outgoing_velocity": outgoing_velocity.copy(),
            "outgoing_spin": outgoing_spin.copy(),
            "impact_velocity_slack_mps": impact_slack,
            "screen_passed": not screen_reasons,
            "screen_reasons": screen_reasons,
        },
    )
    object.__setattr__(
        fit,
        "_physical_diagnostics",
        {
            "minimum_height_m": minimum_height,
            "continuity_m": 0.0,
            "incoming_vertical_speed_ms": float(parameters[2]),
            "speed_ratio": float(np.linalg.norm(outgoing_velocity))
            / max(float(np.linalg.norm(parameters[:3])), 1e-9),
            "impact_velocity_slack_mps": impact_slack,
        },
    )
    return fit


def fit_ballistic_bounce_node_shot(
    index: int,
    contacts: list[dict],
    ball: dict[int, np.ndarray],
    players: dict,
    camera: FrameCamera,
    fps: float,
    surface: str,
    bounce_anchor: dict,
    max_nfev: int,
) -> ShotFit | None:
    """Numerically stable exact-node fallback with bounded per-arc acceleration."""
    start_frame = float(contacts[index]["frame"])
    end_frame = float(contacts[index + 1]["frame"])
    bounce_frame = float(bounce_anchor["frame"])
    if not start_frame < bounce_frame < end_frame:
        return None
    frames = np.array(
        sorted(frame for frame in ball if math.ceil(start_frame) <= frame <= math.floor(end_frame)),
        dtype=int,
    )
    if len(frames) < MIN_OBS:
        return None
    observed = np.stack([ball[int(frame)] for frame in frames])
    projections = [camera.p_at(frame) for frame in frames]
    node = np.asarray(bounce_anchor["x"], float)
    relative = (frames - bounce_frame) / fps
    start_player = nearest(players.get(contacts[index]["side"], {}), start_frame)
    end_player = nearest(players.get(contacts[index + 1]["side"], {}), end_frame)

    def states(parameters: np.ndarray, times: np.ndarray = relative):
        incoming_velocity = parameters[:3]
        outgoing_velocity = parameters[3:6]
        incoming_acceleration = np.r_[parameters[6:8], -9.81 + parameters[8]]
        outgoing_acceleration = np.r_[parameters[9:11], -9.81 + parameters[11]]
        mask = times < 0
        positions = np.empty((len(times), 3))
        velocities = np.empty((len(times), 3))
        for selected, velocity, acceleration in (
            (mask, incoming_velocity, incoming_acceleration),
            (~mask, outgoing_velocity, outgoing_acceleration),
        ):
            elapsed = times[selected, None]
            positions[selected] = node + velocity * elapsed + 0.5 * acceleration * elapsed**2
            velocities[selected] = velocity + acceleration * elapsed
        return positions, velocities, incoming_acceleration, outgoing_acceleration

    def residual(parameters: np.ndarray):
        positions, _, _, _ = states(parameters)
        pixels = np.asarray(
            [project_one(projection, xyz) for projection, xyz in zip(projections, positions)]
        )
        pixels = np.nan_to_num(pixels, nan=1e4, posinf=1e4, neginf=-1e4)
        expected_out, _, _ = bounce_velocity(parameters[:3], np.zeros(3), surface)
        values = [((pixels - observed) / PIXEL_SIGMA).ravel()]
        values.append((parameters[3:6] - expected_out) / BOUNCE_NODE_IMPACT_SLACK_SIGMA_MPS)
        values.append(parameters[6:12] / np.array([3.0, 3.0, 2.5, 3.0, 3.0, 2.5]))
        values.append(np.maximum(0.0, R_BALL - positions[:, 2]) / GROUND_TOLERANCE_M)
        boundary_positions, _, _, _ = states(
            parameters,
            np.array(
                [
                    (start_frame - bounce_frame) / fps,
                    (end_frame - bounce_frame) / fps,
                ]
            ),
        )
        if start_player is not None:
            values.append(
                np.array(
                    [
                        max(
                            0.0,
                            np.linalg.norm(boundary_positions[0, :2] - start_player)
                            - MAX_CONTACT_REACH_M,
                        )
                        / 0.05
                    ]
                )
            )
        if end_player is not None and not contacts[index + 1].get("terminal"):
            values.append(
                np.array(
                    [
                        max(
                            0.0,
                            np.linalg.norm(boundary_positions[1, :2] - end_player)
                            - MAX_CONTACT_REACH_M,
                        )
                        / 0.05
                    ]
                )
            )
        return np.concatenate(values)

    starts = [
        np.array([0.0, sign * speed, -5.0, 0.0, sign * 0.65 * speed, 4.0, *([0.0] * 6)])
        for speed in (20.0, 35.0)
        for sign in (-1.0, 1.0)
    ]
    lower = np.array(
        [-70, -70, -40, -70, -70, 0.05, -8, -8, -8, -8, -8, -8],
        float,
    )
    upper = np.array(
        [70, 70, -0.05, 70, 70, 40, 8, 8, 8, 8, 8, 8],
        float,
    )
    best = None
    for initial in starts:
        try:
            result = least_squares(
                residual,
                initial,
                bounds=(lower, upper),
                loss="soft_l1",
                f_scale=2.0,
                x_scale="jac",
                max_nfev=max(80, max_nfev),
            )
        except (ValueError, FloatingPointError):
            continue
        positions, velocities, incoming_acceleration, outgoing_acceleration = states(result.x)
        pixels = np.asarray(
            [project_one(projection, xyz) for projection, xyz in zip(projections, positions)]
        )
        pixels = np.nan_to_num(pixels, nan=1e4, posinf=1e4, neginf=-1e4)
        rms = float(np.sqrt(np.mean(np.sum((pixels - observed) ** 2, axis=1))))
        expected_out, _, regime = bounce_velocity(result.x[:3], np.zeros(3), surface)
        impact_slack = float(np.linalg.norm(result.x[3:6] - expected_out))
        score = rms + impact_slack / BOUNCE_NODE_IMPACT_SLACK_SIGMA_MPS
        if best is None or score < best[0]:
            best = (
                score,
                result.x,
                rms,
                positions,
                velocities,
                incoming_acceleration,
                outgoing_acceleration,
                impact_slack,
                regime,
            )
    if best is None:
        return None
    (
        _,
        parameters,
        rms,
        positions,
        velocities,
        incoming_acceleration,
        outgoing_acceleration,
        impact_slack,
        regime,
    ) = best
    start_position, start_velocity, _, _ = states(
        parameters,
        np.array([(start_frame - bounce_frame) / fps]),
    )
    start_distance = (
        float(np.linalg.norm(start_position[0, :2] - start_player))
        if start_player is not None
        else 0.0
    )
    end_position, _, _, _ = states(
        parameters,
        np.array([(end_frame - bounce_frame) / fps]),
    )
    end_distance = (
        float(np.linalg.norm(end_position[0, :2] - end_player))
        if end_player is not None and not contacts[index + 1].get("terminal")
        else 0.0
    )
    minimum_height = float(np.min(positions[:, 2]))
    screen_reasons = []
    if rms > MAX_BOUNCE_NODE_RMS_PX:
        screen_reasons.append("high_reprojection")
    if impact_slack > MAX_BOUNCE_NODE_IMPACT_SLACK_MPS:
        screen_reasons.append("impact_slack")
    if minimum_height < R_BALL - GROUND_TOLERANCE_M:
        screen_reasons.append("below_court")
    if start_distance > MAX_BOUNCE_NODE_RECOVERY_REACH_M:
        screen_reasons.append("start_outside_recovery_reach")
    if end_distance > MAX_BOUNCE_NODE_RECOVERY_REACH_M:
        screen_reasons.append("end_outside_recovery_reach")
    bounce = {
        "frame": bounce_frame,
        "x": node.copy(),
        "v_in": parameters[:3].copy(),
        "v_out": parameters[3:6].copy(),
        "w_in": np.zeros(3),
        "w_out": np.zeros(3),
        "regime": regime,
    }
    fit = ShotFit(
        index,
        index + 1,
        np.r_[start_position[0], start_velocity[0], np.zeros(3)],
        rms,
        len(frames),
        positions,
        velocities,
        frames,
        [bounce],
        2,
    )
    object.__setattr__(fit, "_f0", start_frame)
    object.__setattr__(fit, "_weighted_rms_px", rms)
    object.__setattr__(fit, "_ws_obs", np.zeros_like(positions))
    object.__setattr__(fit, "_fixed_bounce_anchor", bounce_anchor)
    object.__setattr__(
        fit,
        "_ballistic_bounce_node",
        {
            "frame": bounce_frame,
            "xyz": node.copy(),
            "incoming_velocity": parameters[:3].copy(),
            "outgoing_velocity": parameters[3:6].copy(),
            "incoming_acceleration": incoming_acceleration.copy(),
            "outgoing_acceleration": outgoing_acceleration.copy(),
            "impact_velocity_slack_mps": impact_slack,
            "screen_passed": not screen_reasons,
            "screen_reasons": screen_reasons,
        },
    )
    object.__setattr__(
        fit,
        "_physical_diagnostics",
        {
            "minimum_height_m": minimum_height,
            "continuity_m": 0.0,
            "incoming_vertical_speed_ms": float(parameters[2]),
            "speed_ratio": float(np.linalg.norm(parameters[3:6]))
            / max(float(np.linalg.norm(parameters[:3])), 1e-9),
            "impact_velocity_slack_mps": impact_slack,
            "model": "exact_node_bounded_piecewise_ballistic",
        },
    )
    return fit


def fit_anchored_ballistic_bounce_node_shot(
    index: int,
    contacts: list[dict],
    ball: dict[int, np.ndarray],
    camera: FrameCamera,
    fps: float,
    surface: str,
    bounce_anchor: dict,
    start_anchor: np.ndarray,
    end_anchor: np.ndarray,
    max_nfev: int,
) -> ShotFit | None:
    """Fit a bounded exact contact-node-contact fallback around a hard bounce."""
    start_frame = float(contacts[index]["frame"])
    end_frame = float(contacts[index + 1]["frame"])
    bounce_frame = float(bounce_anchor["frame"])
    if not start_frame < bounce_frame < end_frame:
        return None
    frames = np.array(
        sorted(frame for frame in ball if math.ceil(start_frame) <= frame <= math.floor(end_frame)),
        dtype=int,
    )
    if len(frames) < MIN_OBS:
        return None
    observed = np.stack([ball[int(frame)] for frame in frames])
    projections = [camera.p_at(frame) for frame in frames]
    node = np.asarray(bounce_anchor["x"], float)
    start_anchor = np.asarray(start_anchor, float)
    end_anchor = np.asarray(end_anchor, float)
    pre_duration = (bounce_frame - start_frame) / fps
    post_duration = (end_frame - bounce_frame) / fps
    relative = (frames - bounce_frame) / fps

    def states(parameters: np.ndarray, times: np.ndarray = relative):
        incoming_acceleration = np.r_[parameters[:2], -9.81 + parameters[2]]
        outgoing_acceleration = np.r_[parameters[3:5], -9.81 + parameters[5]]
        incoming_velocity = (
            node - start_anchor
        ) / pre_duration + 0.5 * incoming_acceleration * pre_duration
        outgoing_velocity = (
            end_anchor - node - 0.5 * outgoing_acceleration * post_duration**2
        ) / post_duration
        mask = times < 0
        positions = np.empty((len(times), 3))
        velocities = np.empty((len(times), 3))
        for selected, velocity, acceleration in (
            (mask, incoming_velocity, incoming_acceleration),
            (~mask, outgoing_velocity, outgoing_acceleration),
        ):
            elapsed = times[selected, None]
            positions[selected] = node + velocity * elapsed + 0.5 * acceleration * elapsed**2
            velocities[selected] = velocity + acceleration * elapsed
        return (
            positions,
            velocities,
            incoming_velocity,
            outgoing_velocity,
            incoming_acceleration,
            outgoing_acceleration,
        )

    def residual(parameters: np.ndarray):
        positions, _, incoming_velocity, outgoing_velocity, _, _ = states(parameters)
        pixels = np.asarray(
            [project_one(projection, xyz) for projection, xyz in zip(projections, positions)]
        )
        pixels = np.nan_to_num(pixels, nan=1e4, posinf=1e4, neginf=-1e4)
        expected_out, _, _ = bounce_velocity(incoming_velocity, np.zeros(3), surface)
        values = [((pixels - observed) / PIXEL_SIGMA).ravel()]
        values.append((outgoing_velocity - expected_out) / BOUNCE_NODE_IMPACT_SLACK_SIGMA_MPS)
        values.append(parameters / np.array([3.0, 3.0, 2.5, 3.0, 3.0, 2.5]))
        values.append(np.maximum(0.0, R_BALL - positions[:, 2]) / GROUND_TOLERANCE_M)
        values.append(np.array([max(0.0, incoming_velocity[2] + MIN_BOUNCE_DESCENT_MS)]))
        values.append(np.array([max(0.0, 0.05 - outgoing_velocity[2])]))
        return np.concatenate(values)

    best = None
    bounds = (np.full(6, -8.0), np.full(6, 8.0))
    for initial in (np.zeros(6), np.array([0.0, 0.0, 2.0, 0.0, 0.0, -2.0])):
        try:
            result = least_squares(
                residual,
                initial,
                bounds=bounds,
                loss="soft_l1",
                f_scale=2.0,
                x_scale="jac",
                max_nfev=max(80, max_nfev),
            )
        except (ValueError, FloatingPointError):
            continue
        (
            positions,
            velocities,
            incoming_velocity,
            outgoing_velocity,
            incoming_acceleration,
            outgoing_acceleration,
        ) = states(result.x)
        pixels = np.asarray(
            [project_one(projection, xyz) for projection, xyz in zip(projections, positions)]
        )
        pixels = np.nan_to_num(pixels, nan=1e4, posinf=1e4, neginf=-1e4)
        rms = float(np.sqrt(np.mean(np.sum((pixels - observed) ** 2, axis=1))))
        expected_out, _, regime = bounce_velocity(incoming_velocity, np.zeros(3), surface)
        impact_slack = float(np.linalg.norm(outgoing_velocity - expected_out))
        score = rms + impact_slack / BOUNCE_NODE_IMPACT_SLACK_SIGMA_MPS
        if best is None or score < best[0]:
            best = (
                score,
                result.x,
                rms,
                positions,
                velocities,
                incoming_velocity,
                outgoing_velocity,
                incoming_acceleration,
                outgoing_acceleration,
                impact_slack,
                regime,
            )
    if best is None:
        return None
    (
        _,
        parameters,
        rms,
        positions,
        velocities,
        incoming_velocity,
        outgoing_velocity,
        incoming_acceleration,
        outgoing_acceleration,
        impact_slack,
        regime,
    ) = best
    minimum_height = float(np.min(positions[:, 2]))
    screen_reasons = []
    if rms > MAX_BOUNCE_NODE_RMS_PX:
        screen_reasons.append("high_reprojection")
    if impact_slack > MAX_BOUNCE_NODE_IMPACT_SLACK_MPS:
        screen_reasons.append("impact_slack")
    if minimum_height < R_BALL - GROUND_TOLERANCE_M:
        screen_reasons.append("below_court")
    if incoming_velocity[2] > -MIN_BOUNCE_DESCENT_MS:
        screen_reasons.append("not_descending_into_bounce")
    if outgoing_velocity[2] < 0.05:
        screen_reasons.append("not_rising_after_bounce")
    start_velocity = incoming_velocity - incoming_acceleration * pre_duration
    bounce = {
        "frame": bounce_frame,
        "x": node.copy(),
        "v_in": incoming_velocity.copy(),
        "v_out": outgoing_velocity.copy(),
        "w_in": np.zeros(3),
        "w_out": np.zeros(3),
        "regime": regime,
    }
    fit = ShotFit(
        index,
        index + 1,
        np.r_[start_anchor, start_velocity, np.zeros(3)],
        rms,
        len(frames),
        positions,
        velocities,
        frames,
        [bounce],
        2,
    )
    object.__setattr__(fit, "_f0", start_frame)
    object.__setattr__(fit, "_weighted_rms_px", rms)
    object.__setattr__(fit, "_ws_obs", np.zeros_like(positions))
    object.__setattr__(fit, "_fixed_bounce_anchor", bounce_anchor)
    object.__setattr__(
        fit,
        "_ballistic_bounce_node",
        {
            "frame": bounce_frame,
            "xyz": node.copy(),
            "incoming_velocity": incoming_velocity.copy(),
            "outgoing_velocity": outgoing_velocity.copy(),
            "incoming_acceleration": incoming_acceleration.copy(),
            "outgoing_acceleration": outgoing_acceleration.copy(),
            "impact_velocity_slack_mps": impact_slack,
            "screen_passed": not screen_reasons,
            "screen_reasons": screen_reasons,
            "model": "exact_contact_node_contact_bounded_piecewise_ballistic",
            "parameters": parameters.copy(),
        },
    )
    object.__setattr__(
        fit,
        "_physical_diagnostics",
        {
            "minimum_height_m": minimum_height,
            "continuity_m": 0.0,
            "incoming_vertical_speed_ms": float(incoming_velocity[2]),
            "speed_ratio": float(np.linalg.norm(outgoing_velocity))
            / max(float(np.linalg.norm(incoming_velocity)), 1e-9),
            "impact_velocity_slack_mps": impact_slack,
            "model": "exact_contact_node_contact_bounded_piecewise_ballistic",
        },
    )
    return fit


def hard_bounce_recovery_score(
    candidate: ShotFit,
    index: int,
    contacts: list[dict],
    ball: dict[int, np.ndarray],
    players: dict,
    camera: FrameCamera,
    fps: float,
    surface: str,
) -> float:
    """Rank exact-node branches by pixels, contact reach, and impact-law deviation."""
    start = contacts[index]
    end = contacts[index + 1]
    start_position, _ = candidate.state(float(start["frame"]), fps, surface)
    end_position, _ = candidate.state(float(end["frame"]), fps, surface)
    endpoint_errors = []
    for position, contact in ((start_position, start), (end_position, end)):
        _, diagnostic = contact_image_residual(position, contact, ball, camera)
        if diagnostic["error_px"] is not None:
            endpoint_errors.append(float(diagnostic["error_px"]))
    start_player = nearest(players.get(start["side"], {}), float(start["frame"]))
    end_player = nearest(players.get(end["side"], {}), float(end["frame"]))
    distances = []
    if start_player is not None:
        distances.append(float(np.linalg.norm(start_position[:2] - start_player)))
    if end_player is not None and not end.get("terminal"):
        distances.append(float(np.linalg.norm(end_position[:2] - end_player)))
    reach_excess = sum(max(0.0, distance - MAX_CONTACT_REACH_M) for distance in distances)
    impact_slack = float(
        getattr(candidate, "_physical_diagnostics", {}).get(
            "impact_velocity_slack_mps",
            0.0,
        )
    )
    endpoint_penalty = 0.5 * max(endpoint_errors, default=0.0)
    return float(candidate.rms_px + endpoint_penalty + 15.0 * reach_excess + impact_slack)


def bounce_recovery_screen_passed(candidate: ShotFit) -> bool:
    """Reject exact-node recovery branches that failed their own absolute screen."""
    for attribute in ("_bounce_node_split", "_ballistic_bounce_node"):
        node = getattr(candidate, attribute, None)
        if node is not None and not bool(node.get("screen_passed", False)):
            return False
    return True


def recover_hard_bounce_shot(
    index: int,
    contacts: list[dict],
    ball: dict[int, np.ndarray],
    players: dict,
    camera: FrameCamera,
    fps: float,
    surface: str,
    bounce_anchor: dict,
    max_nfev: int,
    *,
    allow_initialization_only: bool = True,
    fixed_start_anchor: np.ndarray | None = None,
    fixed_end_anchor: np.ndarray | None = None,
) -> ShotFit | None:
    """Recover a missing flight directly from its contacts and exact ground node."""
    start = contacts[index]
    end = contacts[index + 1]
    start_xyz, _, _ = contact_prior(start, ball, players, camera)
    end_xyz, _, _ = contact_prior(end, ball, players, camera)
    if start_xyz is None or end_xyz is None:
        return None
    start_ray, _ = contact_ray_anchor(start, start_xyz, ball, players, camera, None)
    end_ray, _ = contact_ray_anchor(end, end_xyz, ball, players, camera, None)
    if start_ray is not None:
        start_xyz = start_ray
    if end_ray is not None:
        end_xyz = end_ray
    if fixed_start_anchor is not None:
        start_xyz = np.asarray(fixed_start_anchor, float)
    if fixed_end_anchor is not None:
        end_xyz = np.asarray(fixed_end_anchor, float)
    theta = initial_theta(
        np.asarray(start_xyz, float),
        np.asarray(end_xyz, float),
        (float(end["frame"]) - float(start["frame"])) / fps,
        {},
    )
    seed = ShotFit(
        index,
        index + 1,
        theta,
        math.inf,
        0,
        np.empty((0, 3)),
        np.empty((0, 3)),
        np.empty(0),
        [],
        0,
    )
    object.__setattr__(seed, "_f0", float(start["frame"]))
    recovered = shoot_shot(
        index,
        contacts,
        ball,
        camera,
        fps,
        surface,
        (np.asarray(start_xyz, float), np.asarray(end_xyz, float)),
        seed,
        max_nfev,
        bounce_anchor=bounce_anchor,
    )
    if not allow_initialization_only:
        return recovered
    anchored_recovery = fit_anchored_ballistic_bounce_node_shot(
        index,
        contacts,
        ball,
        camera,
        fps,
        surface,
        bounce_anchor,
        np.asarray(start_xyz, float),
        np.asarray(end_xyz, float),
        max_nfev,
    )
    ballistic_recovery = fit_ballistic_bounce_node_shot(
        index,
        contacts,
        ball,
        players,
        camera,
        fps,
        surface,
        bounce_anchor,
        max_nfev,
    )
    node_recovery = fit_bounce_node_shot(
        index,
        contacts,
        ball,
        players,
        camera,
        fps,
        surface,
        bounce_anchor,
        seed,
        max_nfev,
    )
    recovery_candidates = [
        candidate
        for candidate in (
            (recovered, anchored_recovery)
            if fixed_start_anchor is not None or fixed_end_anchor is not None
            else (recovered, anchored_recovery, ballistic_recovery, node_recovery)
        )
        if candidate is not None and bounce_recovery_screen_passed(candidate)
    ]
    if recovery_candidates:
        return min(
            recovery_candidates,
            key=lambda candidate: hard_bounce_recovery_score(
                candidate,
                index,
                contacts,
                ball,
                players,
                camera,
                fps,
                surface,
            ),
        )
    soft_endpoint = fit_fixed_start_terminal_segment(
        float(start["frame"]),
        float(end["frame"]),
        np.asarray(start_xyz, float),
        ball,
        camera,
        fps,
        surface,
        seed,
        max_nfev,
        bounce_anchor,
    )
    if soft_endpoint is not None and soft_endpoint.rms_px <= 20.0:
        soft_endpoint.start = index
        soft_endpoint.end = index + 1
        object.__setattr__(soft_endpoint, "_soft_endpoint_recovery", True)
        return soft_endpoint
    frames = np.array(
        sorted(
            frame
            for frame in ball
            if math.ceil(float(start["frame"])) <= frame <= math.floor(float(end["frame"]))
        ),
        dtype=int,
    )
    if len(frames) < MIN_OBS:
        return None
    try:
        xs, vs, ws, bounces, continuity = simulate_fixed_bounce_with_spin(
            theta,
            float(start["frame"]),
            frames,
            fps,
            surface,
            bounce_anchor,
        )
    except (ValueError, FloatingPointError):
        return None
    pixels = np.array([project_one(camera.p_at(frame), x) for frame, x in zip(frames, xs)])
    observed = np.stack([ball[int(frame)] for frame in frames])
    rms = float(np.sqrt(np.mean(np.sum((pixels - observed) ** 2, axis=1))))
    initializer = ShotFit(
        index,
        index + 1,
        theta,
        rms,
        len(frames),
        xs,
        vs,
        frames,
        bounces,
        0,
    )
    object.__setattr__(initializer, "_f0", float(start["frame"]))
    object.__setattr__(initializer, "_weighted_rms_px", rms)
    object.__setattr__(initializer, "_ws_obs", ws)
    object.__setattr__(initializer, "_fixed_bounce_anchor", bounce_anchor)
    object.__setattr__(initializer, "_initialization_only", True)
    object.__setattr__(
        initializer,
        "_physical_diagnostics",
        {
            "minimum_height_m": float(np.min(xs[:, 2])),
            "continuity_m": float(np.linalg.norm(continuity)),
            "incoming_vertical_speed_ms": None,
            "speed_ratio": None,
        },
    )
    return initializer


def initialize_missing_shot(
    index: int,
    contacts: list[dict],
    ball: dict[int, np.ndarray],
    players: dict,
    camera: FrameCamera,
    fps: float,
    surface: str,
) -> ShotFit | None:
    """Create a non-emittable seed so global refinement can attempt a missing flight."""
    start = contacts[index]
    end = contacts[index + 1]
    start_xyz, _, _ = contact_prior(start, ball, players, camera)
    end_xyz, _, _ = contact_prior(end, ball, players, camera)
    if start_xyz is None or end_xyz is None:
        return None
    frames = np.array(
        sorted(
            frame
            for frame in ball
            if math.ceil(float(start["frame"])) <= frame <= math.floor(float(end["frame"]))
        ),
        dtype=int,
    )
    if len(frames) < MIN_OBS:
        return None
    theta = initial_theta(
        np.asarray(start_xyz, float),
        np.asarray(end_xyz, float),
        (float(end["frame"]) - float(start["frame"])) / fps,
        {},
    )
    try:
        xs, vs, ws, bounces = simulate_with_spin(
            theta,
            float(start["frame"]),
            frames,
            fps,
            surface,
        )
    except (ValueError, FloatingPointError):
        return None
    pixels = np.array([project_one(camera.p_at(frame), x) for frame, x in zip(frames, xs)])
    observed = np.stack([ball[int(frame)] for frame in frames])
    rms = float(np.sqrt(np.mean(np.sum((pixels - observed) ** 2, axis=1))))
    initializer = ShotFit(
        index,
        index + 1,
        theta,
        rms,
        len(frames),
        xs,
        vs,
        frames,
        bounces,
        0,
    )
    object.__setattr__(initializer, "_f0", float(start["frame"]))
    object.__setattr__(initializer, "_weighted_rms_px", rms)
    object.__setattr__(initializer, "_ws_obs", ws)
    object.__setattr__(initializer, "_initialization_only", True)
    object.__setattr__(
        initializer,
        "_physical_diagnostics",
        {
            "minimum_height_m": float(np.min(xs[:, 2])),
            "continuity_m": None,
            "incoming_vertical_speed_ms": None,
            "speed_ratio": None,
        },
    )
    return initializer


def shoot_shot(
    index: int,
    contacts: list[dict],
    ball: dict[int, np.ndarray],
    camera: FrameCamera,
    fps: float,
    surface: str,
    anchors: tuple[np.ndarray, np.ndarray],
    seed: ShotFit,
    max_nfev: int,
    bounce_anchor: dict | None = None,
) -> ShotFit | None:
    """Boundary-value refit: shared contact endpoints are exact, physics fills the path.

    Once adjacent one-sided fits have proposed a common contact location, the remaining
    problem is a shooting problem.  For each plausible topspin value, solve the outgoing
    velocity which lands the integrated flight exactly on the next shared contact; select
    spin using all intervening image observations.  Unlike a soft endpoint penalty, this
    cannot trade a metre-scale junction jump for a few pixels of reprojection error.
    """
    c0, c1 = contacts[index], contacts[index + 1]
    f0, f1 = c0["frame"], c1["frame"]
    frames = np.array(sorted(f for f in ball if math.ceil(f0) <= f <= math.floor(f1)), int)
    if len(frames) < MIN_OBS:
        return None
    uv = np.stack([ball[f] for f in frames])
    cameras = [camera.p_at(f) for f in frames]
    p0, p1 = (np.asarray(point, float) for point in anchors)
    seed_theta = full_spin_theta(seed.theta)
    spin_candidates = [
        seed_theta[6:9],
        np.zeros(3),
        np.array([-3.0, 0.0, 0.0]),
        np.array([3.0, 0.0, 0.0]),
        np.array([1.0, 3.0, 0.0]),
        np.array([1.0, -3.0, 0.0]),
    ]
    spin_candidates = list(
        {tuple(np.clip(candidate, [-5, -5, -3], [5, 5, 3])) for candidate in spin_candidates}
    )
    velocity_starts = [seed.theta[3:6]]
    duration = (f1 - f0) / fps
    direct = (p1 - p0) / max(duration, 0.2)
    for vz in (5.0, 12.0):
        velocity_starts.append(np.array([direct[0], direct[1], vz]))
    best = None
    hard_bounce = (
        bounce_anchor if bounce_anchor is not None and bounce_anchor.get("hard_geometry") else None
    )
    for spin in spin_candidates:
        for velocity0 in velocity_starts:

            def endpoint_residual(velocity):
                theta = np.r_[p0, velocity, spin]
                if hard_bounce is not None:
                    end, _, bounces, _ = simulate_fixed_bounce(
                        theta,
                        f0,
                        np.array([f1]),
                        fps,
                        surface,
                        hard_bounce,
                    )
                else:
                    end, _, bounces = simulate(theta, f0, np.array([f1]), fps, surface)
                outside = sum(
                    max(0.0, -b["x"][0])
                    + max(0.0, b["x"][0] - 10.97)
                    + max(0.0, -b["x"][1])
                    + max(0.0, b["x"][1] - 23.77)
                    for b in bounces
                )
                # Three endpoint equations plus a fixed-length physical validity term.
                return np.r_[(end[0] - p1) / 0.02, outside]

            try:
                solution = least_squares(
                    endpoint_residual,
                    np.clip(velocity0, -65, 65),
                    bounds=(-70 * np.ones(3), 70 * np.ones(3)),
                    x_scale="jac",
                    max_nfev=max_nfev,
                )
            except (ValueError, FloatingPointError):
                continue
            theta = np.r_[p0, solution.x, spin]
            if hard_bounce is not None:
                xs, vs, ws, bounces, bounce_continuity = simulate_fixed_bounce_with_spin(
                    theta,
                    f0,
                    frames,
                    fps,
                    surface,
                    hard_bounce,
                )
                end, _, _, _ = simulate_fixed_bounce(
                    theta,
                    f0,
                    np.array([f1]),
                    fps,
                    surface,
                    hard_bounce,
                )
                dense_frames = (
                    np.arange(
                        math.ceil(f0 * 4.0),
                        math.floor(f1 * 4.0) + 1,
                    )
                    / 4.0
                )
                dense_xs, _, _, dense_bounces, dense_continuity = simulate_fixed_bounce_with_spin(
                    theta,
                    f0,
                    dense_frames,
                    fps,
                    surface,
                    hard_bounce,
                )
                diagnostics = fixed_bounce_diagnostics(
                    dense_xs,
                    dense_bounces[0],
                    dense_continuity,
                )
                if not diagnostics["valid"]:
                    continue
            else:
                xs, vs, ws, bounces = simulate_with_spin(
                    theta,
                    f0,
                    frames,
                    fps,
                    surface,
                )
                end, _, _ = simulate(theta, f0, np.array([f1]), fps, surface)
                diagnostics = {
                    "minimum_height_m": float(np.min(xs[:, 2])),
                    "continuity_m": None,
                    "incoming_vertical_speed_ms": None,
                    "speed_ratio": max(
                        (
                            float(np.linalg.norm(row["v_out"]))
                            / max(float(np.linalg.norm(row["v_in"])), 1e-9)
                            for row in bounces
                        ),
                        default=None,
                    ),
                }
            endpoint_gap = float(np.linalg.norm(end[0] - p1))
            if endpoint_gap > 0.02 or np.linalg.norm(solution.x) > MAX_SPEED_MS:
                continue
            pixels = np.array([project_one(P, x) for P, x in zip(cameras, xs)])
            residuals = np.linalg.norm(pixels - uv, axis=1)
            # The decoded WASB path contains isolated identity switches.  Model selection
            # uses a declared 90% trimmed RMS; all-observation RMS is still reported.
            keep = residuals <= np.quantile(residuals, 0.9)
            trimmed_rms = float(np.sqrt(np.mean(residuals[keep] ** 2)))
            rms = float(np.sqrt(np.mean(residuals**2)))
            outside_m = sum(
                max(0.0, -b["x"][0])
                + max(0.0, b["x"][0] - 10.97)
                + max(0.0, -b["x"][1])
                + max(0.0, b["x"][1] - 23.77)
                for b in bounces
            )
            bounce_cost = 0.0
            if bounce_anchor is not None and hard_bounce is None:
                if not bounces:
                    bounce_cost = 100.0
                else:
                    bounce_cost = abs(bounces[0]["frame"] - bounce_anchor["frame"])
                    bounce_cost += float(
                        np.linalg.norm(bounces[0]["x"][:2] - bounce_anchor["x"][:2])
                        / float(bounce_anchor.get("sigma_xy_m", 0.15))
                    )
            score = trimmed_rms + 10.0 * outside_m + 3.0 * bounce_cost
            if best is None or score < best[0]:
                best = score, theta, rms, xs, vs, ws, bounces, diagnostics
    if best is None:
        return None
    _, theta, rms, xs, vs, ws, bounces, diagnostics = best
    fit = ShotFit(index, index + 1, theta, rms, len(frames), xs, vs, frames, bounces, 2)
    object.__setattr__(fit, "_f0", f0)
    object.__setattr__(fit, "_weighted_rms_px", rms)
    object.__setattr__(fit, "_ws_obs", ws)
    object.__setattr__(
        fit,
        "_physical_diagnostics",
        diagnostics,
    )
    if hard_bounce is not None:
        object.__setattr__(fit, "_fixed_bounce_anchor", hard_bounce)
    return fit


def connected_shot_chains(fits: dict[int, ShotFit]) -> list[list[int]]:
    chains: list[list[int]] = []
    for index in sorted(fits):
        if not chains or index != chains[-1][-1] + 1:
            chains.append([index])
        else:
            chains[-1].append(index)
    return chains


def jointly_refine_chain(
    indices: list[int],
    contacts: list[dict],
    ball,
    players,
    camera: FrameCamera,
    fps: float,
    surface: str,
    initial_fits: dict[int, ShotFit],
    initial_positions: dict[int, np.ndarray],
    max_nfev: int,
    bounce_anchors: list[dict],
    contact_geometry=None,
    observation_weights: dict[int, float] | None = None,
    spin_priors: dict[int, dict] | None = None,
):
    """Sparse bundle adjustment over shared contacts and every intervening flight.

    A contact position appears once in the parameter vector, so incoming and outgoing
    shots cannot choose different camera-ray depths. Each flight retains its own
    outgoing velocity and three-axis spin. The Jacobian sparsity pattern tells SciPy that a shot
    only touches its two endpoints, keeping long-rally finite differencing tractable.
    """
    contact_ids = list(range(indices[0], indices[-1] + 2))
    contact_slot = {contact_id: slot for slot, contact_id in enumerate(contact_ids)}
    shot_slot = {shot_id: slot for slot, shot_id in enumerate(indices)}
    n_pos = 3 * len(contact_ids)

    parts = [np.concatenate([initial_positions[i] for i in contact_ids])]
    parts.append(
        np.concatenate(
            [
                np.r_[
                    full_spin_theta(initial_fits[i].theta)[3:6],
                    full_spin_theta(initial_fits[i].theta)[6:9],
                ]
                for i in indices
            ]
        )
    )
    params0 = np.concatenate(parts)
    contact_lower = []
    contact_upper = []
    for contact_id in contact_ids:
        contact = contacts[contact_id]
        player = nearest(players.get(contact["side"], {}), contact["frame"])
        if player is None or contact.get("terminal"):
            contact_lower.extend([-3.0, -8.0, 0.15])
            contact_upper.extend([14.0, 32.0, 3.5])
            continue
        reach = 1.6 if contact.get("phase") == "serve" else MAX_CONTACT_REACH_M
        reach_lower, reach_upper = contact_xy_bounds(player, reach)
        minimum_z = 1.0 if contact.get("phase") == "serve" else 0.15
        contact_lower.extend([*reach_lower, minimum_z])
        contact_upper.extend([*reach_upper, 3.5])
    lower = np.r_[contact_lower, np.tile([-70, -70, -40, -5, -5, -3], len(indices))]
    upper = np.r_[contact_upper, np.tile([70, 70, 40, 5, 5, 3], len(indices))]

    shot_data = {}
    for index in indices:
        f0, f1 = contacts[index]["frame"], contacts[index + 1]["frame"]
        frames = np.array(sorted(f for f in ball if math.ceil(f0) <= f <= math.floor(f1)), int)
        weights = np.array(
            [
                float(
                    np.clip(
                        (observation_weights or {}).get(int(frame), 1.0),
                        0.05,
                        1.0,
                    )
                )
                for frame in frames
            ]
        )
        shot_data[index] = (
            f0,
            f1,
            frames,
            np.stack([ball[f] for f in frames]),
            [camera.p_at(f) for f in frames],
            bounce_for_shot(bounce_anchors, f0, f1),
            weights,
        )

    def position(params, contact_id):
        start = 3 * contact_slot[contact_id]
        return params[start : start + 3]

    def state_params(params, shot_id):
        start = n_pos + 6 * shot_slot[shot_id]
        return params[start : start + 6]

    def residual(params):
        values = []
        for index in indices:
            f0, f1, frames, uv, cameras, bounce_anchor, weights = shot_data[index]
            state = state_params(params, index)
            theta = np.r_[position(params, index), state]
            query = np.r_[frames.astype(float), f1]
            hard_bounce = (
                bounce_anchor
                if bounce_anchor is not None and bounce_anchor.get("hard_geometry")
                else None
            )
            if hard_bounce is not None:
                xs, _, bounces, bounce_continuity = simulate_fixed_bounce(
                    theta,
                    f0,
                    query,
                    fps,
                    surface,
                    hard_bounce,
                )
            else:
                xs, _, bounces = simulate(theta, f0, query, fps, surface)
                bounce_continuity = None
            pixels = np.array([project_one(P, x) for P, x in zip(cameras, xs[:-1])])
            values.append((((pixels - uv) / PIXEL_SIGMA) * np.sqrt(weights)[:, None]).ravel())
            values.append((xs[-1] - position(params, index + 1)) / 0.01)
            outside = sum(
                max(0.0, -b["x"][0])
                + max(0.0, b["x"][0] - 10.97)
                + max(0.0, -b["x"][1])
                + max(0.0, b["x"][1] - 23.77)
                for b in bounces
            )
            values.append(
                np.array(
                    [
                        outside,
                        max(0.0, np.linalg.norm(state[:3]) - MAX_SPEED_MS) / 2,
                        0.15 * abs(state[3]),
                        0.15 * abs(state[4]),
                        0.20 * abs(state[5]),
                    ]
                )
            )
            ground_residual = np.maximum(0.0, R_BALL - xs[:, 2]) / GROUND_TOLERANCE_M
            impact_direction = (
                max(0.0, float(bounces[0]["v_in"][2]) + MIN_BOUNCE_DESCENT_MS)
                / MIN_BOUNCE_DESCENT_MS
                if hard_bounce is not None and bounces
                else 0.0
            )
            values.append(np.r_[ground_residual, impact_direction])
            spin_prior = (spin_priors or {}).get(index)
            if spin_prior is not None:
                fitted_rpm = state[3:6] * 100.0 * 60.0 / (2.0 * math.pi)
                values.append(
                    (
                        fitted_rpm
                        - np.asarray(
                            spin_prior.get(
                                "spin_center_components_rpm",
                                [spin_prior["spin_center_rpm"], 0.0, 0.0],
                            ),
                            float,
                        )
                    )
                    / np.asarray(
                        spin_prior.get(
                            "spin_sigma_components_rpm",
                            [spin_prior["spin_sigma_rpm"], 1200.0, 800.0],
                        ),
                        float,
                    )
                )
            if bounce_anchor is not None:
                if hard_bounce is not None:
                    values.append(bounce_continuity / FIXED_BOUNCE_CONTINUITY_SIGMA_M)
                elif bounces:
                    values.append(
                        np.r_[
                            (bounces[0]["x"][:2] - bounce_anchor["x"][:2])
                            / float(bounce_anchor.get("sigma_xy_m", 0.15)),
                            (bounces[0]["frame"] - bounce_anchor["frame"])
                            / float(bounce_anchor.get("sigma_frame", 0.75)),
                        ]
                    )
                else:
                    values.append(np.full(3, 50.0))
        for contact_id in contact_ids:
            contact = contacts[contact_id]
            point = position(params, contact_id)
            image_residual, _ = contact_image_residual(
                point,
                contact,
                ball,
                camera,
                observation_weights,
            )
            values.append(image_residual)
            if contact.get("terminal"):
                values.append(np.zeros(3))
                continue
            geometry = (contact_geometry or {}).get(contact_id)
            if geometry is not None:
                values.append((point - geometry["center"]) / np.maximum(geometry["sigma"], 0.05))
            else:
                player = nearest(players.get(contact["side"], {}), contact["frame"])
                if player is None:
                    values.append(np.zeros(2))
                else:
                    values.append((point[:2] - player) / PLAYER_SIGMA_M)
                target_z = 2.65 if contact["phase"] == "serve" else 1.5
                sigma_z = 0.75 if contact["phase"] == "serve" else 1.0
                values.append(np.array([(point[2] - target_z) / sigma_z]))
        return np.concatenate(values)

    # Conservative block sparsity: every residual from a shot may depend on its start,
    # end, and velocity/spin; every contact image/player prior depends only on one contact.
    n_residual = len(residual(params0))
    sparsity = lil_matrix((n_residual, len(params0)), dtype=int)
    row = 0
    for index in indices:
        n = len(shot_data[index][2])
        width = (
            2 * n
            + 3
            + 5
            + n
            + 2
            + (3 if index in (spin_priors or {}) else 0)
            + (3 if shot_data[index][5] is not None else 0)
        )
        columns = []
        for contact_id in (index, index + 1):
            start = 3 * contact_slot[contact_id]
            columns.extend(range(start, start + 3))
        start = n_pos + 6 * shot_slot[index]
        columns.extend(range(start, start + 6))
        sparsity[row : row + width, columns] = 1
        row += width
    for contact_id in contact_ids:
        start = 3 * contact_slot[contact_id]
        sparsity[row : row + 5, start : start + 3] = 1
        row += 5
    if row != n_residual:
        raise AssertionError(f"sparsity rows {row} != residual rows {n_residual}")

    result = least_squares(
        residual,
        np.clip(params0, lower + 1e-5, upper - 1e-5),
        bounds=(lower, upper),
        jac_sparsity=sparsity.tocsr(),
        loss="soft_l1",
        f_scale=2.0,
        x_scale="jac",
        max_nfev=max_nfev,
    )
    positions = {i: position(result.x, i).copy() for i in contact_ids}
    fits = {}
    for index in indices:
        f0, _, frames, uv, cameras, _, weights = shot_data[index]
        theta = np.r_[positions[index], state_params(result.x, index)]
        bounce_anchor = shot_data[index][5]
        hard_bounce = (
            bounce_anchor
            if bounce_anchor is not None and bounce_anchor.get("hard_geometry")
            else None
        )
        if hard_bounce is not None:
            xs, vs, ws, bounces, bounce_continuity = simulate_fixed_bounce_with_spin(
                theta,
                f0,
                frames,
                fps,
                surface,
                hard_bounce,
            )
            dense_frames = (
                np.arange(
                    math.ceil(f0 * 4.0),
                    math.floor(float(shot_data[index][1]) * 4.0) + 1,
                )
                / 4.0
            )
            dense_xs, _, _, dense_bounces, dense_continuity = simulate_fixed_bounce_with_spin(
                theta,
                f0,
                dense_frames,
                fps,
                surface,
                hard_bounce,
            )
            diagnostics = fixed_bounce_diagnostics(
                dense_xs,
                dense_bounces[0],
                dense_continuity,
            )
            if not diagnostics["valid"]:
                continue
        else:
            xs, vs, ws, bounces = simulate_with_spin(
                theta,
                f0,
                frames,
                fps,
                surface,
            )
            diagnostics = {
                "minimum_height_m": float(np.min(xs[:, 2])),
                "continuity_m": None,
                "incoming_vertical_speed_ms": None,
                "speed_ratio": max(
                    (
                        float(np.linalg.norm(row["v_out"]))
                        / max(float(np.linalg.norm(row["v_in"])), 1e-9)
                        for row in bounces
                    ),
                    default=None,
                ),
            }
        pixels = np.array([project_one(P, x) for P, x in zip(cameras, xs)])
        squared = np.sum((pixels - uv) ** 2, axis=1)
        rms = float(np.sqrt(np.mean(squared)))
        weighted_rms = float(np.sqrt(np.average(squared, weights=weights)))
        fit = ShotFit(index, index + 1, theta, rms, len(frames), xs, vs, frames, bounces, 2)
        object.__setattr__(fit, "_f0", f0)
        object.__setattr__(fit, "_weighted_rms_px", weighted_rms)
        object.__setattr__(fit, "_ws_obs", ws)
        object.__setattr__(fit, "_physical_diagnostics", diagnostics)
        if hard_bounce is not None:
            object.__setattr__(fit, "_fixed_bounce_anchor", hard_bounce)
        fits[index] = fit
    return positions, fits


def jointly_refine_all(
    contacts,
    ball,
    players,
    camera,
    fps,
    surface,
    initial_fits,
    initial_positions,
    max_nfev,
    bounce_anchors,
    contact_geometry=None,
    observation_weights=None,
    spin_priors=None,
):
    positions = dict(initial_positions)
    preserved = {
        index: fit
        for index, fit in initial_fits.items()
        if getattr(fit, "_bounce_node_split", None) is not None
        or getattr(fit, "_ballistic_bounce_node", None) is not None
    }
    refinable = {index: fit for index, fit in initial_fits.items() if index not in preserved}
    fits = dict(preserved)
    for indices in connected_shot_chains(refinable):
        chain_positions, chain_fits = jointly_refine_chain(
            indices,
            contacts,
            ball,
            players,
            camera,
            fps,
            surface,
            refinable,
            initial_positions,
            max_nfev,
            bounce_anchors,
            contact_geometry,
            observation_weights,
            spin_priors,
        )
        positions.update(chain_positions)
        fits.update(chain_fits)
    return positions, fits


def state_at(fit: ShotFit, frame: float, fps: float, surface: str):
    return fit.state(frame, fps, surface)


def fuse_contacts(
    contacts: list[dict],
    fits: dict[int, ShotFit],
    players,
    ball,
    camera: FrameCamera,
    fps: float,
    surface: str,
):
    fused: dict[int, np.ndarray] = {}
    gaps: dict[int, float] = {}
    for i, contact in enumerate(contacts):
        estimates, weights = [], []
        if i - 1 in fits:
            x, _ = state_at(fits[i - 1], contact["frame"], fps, surface)
            estimates.append(x)
            weights.append(1 / max(fits[i - 1].rms_px, 1) ** 2)
        if i in fits:
            estimates.append(fits[i].theta[:3])
            weights.append(1 / max(fits[i].rms_px, 1) ** 2)
        if len(estimates) == 2:
            gaps[i] = float(np.linalg.norm(estimates[0] - estimates[1]))
        if estimates:
            point = np.average(estimates, axis=0, weights=weights)
        else:
            point, _, _ = contact_prior(contact, ball, players, camera)
        player = nearest(players.get(contact["side"], {}), contact["frame"])
        if player is not None and not contact.get("terminal"):
            # The path estimates choose the reach direction; the player prior only caps
            # physically impossible depth/width displacement.
            delta = point[:2] - player
            norm = float(np.linalg.norm(delta))
            if norm > 1.8:
                point[:2] = player + delta * 1.8 / norm
        point[2] = float(np.clip(point[2], 0.15, 3.5))
        fused[i] = point
    return fused, gaps


def implied_racket(v_in: np.ndarray | None, v_out: np.ndarray | None):
    if v_in is None or v_out is None:
        return None
    impulse = v_out - v_in
    norm = float(np.linalg.norm(impulse))
    if norm < 1e-6:
        return None
    normal = impulse / norm
    e = impact.E_N_RACKET
    racket_normal_speed = (float(np.dot(v_out, normal)) + e * float(np.dot(v_in, normal))) / (1 + e)
    yaw = math.degrees(math.atan2(normal[1], normal[0]))
    pitch = math.degrees(math.atan2(normal[2], math.hypot(normal[0], normal[1])))
    return normal, racket_normal_speed, yaw, pitch


def _valid_flight_endpoint(
    contacts: list[dict], m: int, direction: int, ball: dict[int, np.ndarray], fps: float
) -> int | None:
    """Nearest contact from m in `direction` (+1 forward / -1 back) that can bound a real
    cross-net flight with m, within the live-gap. Two tiers:

    - PREFERRED: the nearest OPPOSITE (or unknown) side contact with >= MIN_OBS observations
      in the span. Intervening same-side contacts (phantom splits of one flight) and too-short
      spans are skipped — the insertion is spanned, not treated as a boundary.
    - FALLBACK (only if no opposite endpoint is reachable): the nearest SAME-side contact with
      enough observations. This is a genuine flight endpoint whose side was mislabelled (e.g. a
      serve whose returner was tagged same-side); using it recovers the real flight. The energy
      gate on the rescued fit rejects low-speed same-side rolls/pre-serve bounces here.

    Returns (endpoint index, is_opposite) or (None, False) if nothing within the gap has
    enough observations. is_opposite marks the preferred tier (a clean cross-net flight);
    False marks the same-side fallback (a mislabelled boundary endpoint, e.g. a serve)."""
    side_m = contacts[m]["side"]
    fm = contacts[m]["frame"]
    fallback = None
    j = m + direction
    while 0 <= j < len(contacts):
        fj = contacts[j]["frame"]
        if abs(fj - fm) > MAX_LIVE_GAP_S * fps:
            break
        lo, hi = (fm, fj) if direction > 0 else (fj, fm)
        nobs = sum(1 for f in ball if math.ceil(lo) <= f <= math.floor(hi))
        if nobs >= MIN_OBS:
            opposite = side_m == "unknown" or contacts[j]["side"] != side_m
            if opposite:
                return j, True
            if fallback is None:
                fallback = j
        j += direction
    return fallback, False


def rescue_abstaining_contacts(
    contacts: list[dict],
    first: dict[int, ShotFit],
    ball,
    players,
    camera: FrameCamera,
    fps: float,
    surface: str,
    max_nfev: int,
) -> dict[int, dict]:
    """One-sided skip-fit for contacts left with NO fittable flight on either adjacent side.

    A contact abstains when both its adjacent spans were refused (same known side) or too
    short to fit — exactly what happens when a spurious contact is INSERTED next to, or on
    the same side as, a real one: it splits the real contact's only cross-net flight into an
    unfittable same-side stub, so the real contact silently vanishes from emission (the
    export-evidence regression: adding one contact drops a NEIGHBOUR). This pass restores the
    real flight by fitting m to the nearest endpoint that can actually bound a cross-net
    flight, spanning the insertion. It only ever ADDS a fit for an otherwise-lost contact and
    never alters an existing fit, so an inserted contact degrades only ITSELF. The one-sided
    straightness gate and impulse prune (knot_prune_loop) then decide which member of a
    same-side cluster is the real strike (bent path) versus a phantom split (straight path).
    """

    def rescue_flight(a: int, b: int) -> ShotFit | None:
        """fit_shot spanning contacts a<b. A mislabelled same-side endpoint pair is a
        cross-net flight whose sides are wrong, so re-fit under both opposite-side anchor
        hypotheses and keep the better-scoring geometry (as the double-unknown path does)."""
        sa, sb = contacts[a]["side"], contacts[b]["side"]
        if sa != "unknown" and sb != "unknown" and sa == sb:
            best = None
            for override in (("near", "far"), ("far", "near")):
                cand = fit_shot(
                    a,
                    contacts,
                    ball,
                    players,
                    camera,
                    fps,
                    surface,
                    max_nfev,
                    end_index=b,
                    side_override=override,
                )
                if cand is not None and (best is None or cand._score < best._score):
                    best = cand
            return best
        return fit_shot(a, contacts, ball, players, camera, fps, surface, max_nfev, end_index=b)

    rescue: dict[int, dict] = {}
    for m in range(len(contacts)):
        if m in first or (m - 1) in first:
            continue  # already has an outgoing or incoming flight
        entry: dict = {"v_in": None, "v_out": None}
        opposite_tier = False
        j, opp_f = _valid_flight_endpoint(contacts, m, +1, ball, fps)
        if j is not None:
            fit = rescue_flight(m, j)
            if fit is not None:
                entry["v_out"] = fit.theta[3:6].copy()
                entry["pos"] = fit.theta[:3].copy()
                opposite_tier = opposite_tier or opp_f
        k, opp_b = _valid_flight_endpoint(contacts, m, -1, ball, fps)
        if k is not None:
            fit = rescue_flight(k, m)
            if fit is not None:
                x, v = state_at(fit, contacts[m]["frame"], fps, surface)
                entry["v_in"] = v.copy()
                entry.setdefault("pos", x.copy())
                opposite_tier = opposite_tier or opp_b
        v = entry["v_out"] if entry["v_out"] is not None else entry["v_in"]
        if v is None:
            continue
        speed = max(
            np.linalg.norm(entry["v_out"]) if entry["v_out"] is not None else 0.0,
            np.linalg.norm(entry["v_in"]) if entry["v_in"] is not None else 0.0,
        )
        # Energy gate. A rescue that reached a genuine OPPOSITE-side endpoint is a clean
        # cross-net rally flight and must look struck (>= AMBIG_SPEED_MIN) — a slow one is a
        # phantom split, not a hit. A same-side FALLBACK endpoint is a mislabelled boundary
        # flight (a serve), whose 2 s bounce arc through wrong-side anchor geometry fits slow
        # even when real; discovery already filtered dead/low-energy breaks and the boundary
        # guard protects it downstream, so only a parked-ball floor applies there.
        floor = AMBIG_SPEED_MIN_MS if opposite_tier else RESCUE_FALLBACK_SPEED_MIN_MS
        if speed >= floor:
            rescue[m] = entry
    return rescue


def write_outputs(spec: MatchSpec, match: str, clips: list[str], all_results: dict, tag: str):
    prefix = os.path.join(spec.output_dir, tag)
    ball_path, contact_path, bounce_path = (
        prefix + ".csv.gz",
        prefix + "_contacts.csv",
        prefix + "_bounces.csv",
    )
    ball_fields = [
        "match",
        "clip",
        "frame",
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "spin_x",
        "spin_y",
        "spin_z",
        "confidence",
        "ci95_x_m",
        "ci95_y_m",
        "ci95_z_m",
        "velocity_ci95_ms",
        "spin_ci95_rad_s",
        "segment",
        "fit_rms_px",
        "source",
    ]
    contact_fields = [
        "match",
        "clip",
        "contact_index",
        "frame",
        "frame_detector",
        "frame_delta",
        "frame_lo",
        "frame_hi",
        "t_lo_s",
        "t_hi_s",
        "timing_source",
        "timing_residual_px",
        "side",
        "phase",
        "status",
        "x",
        "y",
        "z",
        "ci95_x_m",
        "ci95_y_m",
        "ci95_z_m",
        "vx_in",
        "vy_in",
        "vz_in",
        "speed_in",
        "vx_out",
        "vy_out",
        "vz_out",
        "speed_out",
        "junction_gap_pass1_m",
        "junction_gap_pass2_m",
        "racket_normal_x",
        "racket_normal_y",
        "racket_normal_z",
        "racket_normal_speed_ms",
        "racket_face_yaw_deg",
        "racket_face_pitch_deg",
        "velocity_ci95_ms",
        "racket_normal_ci95_deg",
        "racket_speed_ci95_ms",
        "rms_in_px",
        "rms_out_px",
        "uncertainty_method",
        "timing_audio_onset",
        "timing_audio_delta_f",
        "timing_audio_abstain",
    ]
    bounce_fields = [
        "match",
        "clip",
        "segment",
        "frame",
        "x",
        "y",
        "z",
        "vx_in",
        "vy_in",
        "vz_in",
        "vx_out",
        "vy_out",
        "vz_out",
        "regime",
        "fit_rms_px",
    ]
    with (
        gzip.open(ball_path, "wt", newline="") as bf,
        open(contact_path, "w", newline="") as cf,
        open(bounce_path, "w", newline="") as xf,
    ):
        bw, cw, xw = (
            csv.DictWriter(bf, ball_fields),
            csv.DictWriter(cf, contact_fields),
            csv.DictWriter(xf, bounce_fields),
        )
        bw.writeheader()
        cw.writeheader()
        xw.writeheader()
        for clip in clips:
            result = all_results.get(clip)
            if not result:
                continue
            contacts, fits, fused, gaps1, gaps2, rescue = result
            witness = load_witness(spec, match)
            audio_onsets = load_audio_onsets(spec, clip)
            for i, fit in sorted(fits.items()):
                frames = np.arange(math.ceil(fit.f0), math.floor(contacts[i + 1]["frame"]) + 1)
                if (
                    i - 1 in fits
                    and frames.size
                    and abs(fit.f0 - round(fit.f0)) < 1e-9
                    and frames[0] == round(fit.f0)
                ):
                    # One Layer-1 row per native frame. At an integer contact the
                    # incoming segment owns the shared frame; Layer 2 carries both v_in/out.
                    frames = frames[1:]
                xs, vs, bounces = simulate(fit.theta, fit.f0, frames, spec.fps, spec.surface)
                w = spin_vector(fit.theta)
                ci = 1.96 * (
                    0.08 + 0.035 * fit.rms_px + 0.15 * max(gaps2.get(i, 0), gaps2.get(i + 1, 0))
                )
                confidence = float(np.clip(math.exp(-fit.rms_px / 6), 0.02, 0.99))
                velocity_ci = 0.5 + 0.12 * fit.rms_px
                spin_ci = 100.0 + 20.0 * fit.rms_px
                for frame, x, v in zip(frames, xs, vs):
                    bw.writerow(
                        dict(
                            match=match,
                            clip=clip,
                            frame=int(frame),
                            x=f"{x[0]:.4f}",
                            y=f"{x[1]:.4f}",
                            z=f"{x[2]:.4f}",
                            vx=f"{v[0]:.4f}",
                            vy=f"{v[1]:.4f}",
                            vz=f"{v[2]:.4f}",
                            spin_x=f"{w[0]:.3f}",
                            spin_y=f"{w[1]:.3f}",
                            spin_z=f"{w[2]:.3f}",
                            confidence=f"{confidence:.4f}",
                            ci95_x_m=f"{ci:.3f}",
                            ci95_y_m=f"{ci:.3f}",
                            ci95_z_m=f"{ci:.3f}",
                            velocity_ci95_ms=f"{velocity_ci:.3f}",
                            spin_ci95_rad_s=f"{spin_ci:.1f}",
                            segment=i,
                            fit_rms_px=f"{fit.rms_px:.3f}",
                            source=tag,
                        )
                    )
                for bounce in bounces:
                    xw.writerow(
                        dict(
                            match=match,
                            clip=clip,
                            segment=i,
                            frame=f"{bounce['frame']:.3f}",
                            x=f"{bounce['x'][0]:.4f}",
                            y=f"{bounce['x'][1]:.4f}",
                            z=f"{bounce['x'][2]:.4f}",
                            vx_in=f"{bounce['v_in'][0]:.4f}",
                            vy_in=f"{bounce['v_in'][1]:.4f}",
                            vz_in=f"{bounce['v_in'][2]:.4f}",
                            vx_out=f"{bounce['v_out'][0]:.4f}",
                            vy_out=f"{bounce['v_out'][1]:.4f}",
                            vz_out=f"{bounce['v_out'][2]:.4f}",
                            regime=bounce["regime"],
                            fit_rms_px=f"{fit.rms_px:.3f}",
                        )
                    )
            for i, contact in enumerate(contacts):
                fit_in, fit_out = fits.get(i - 1), fits.get(i)
                v_in = (
                    state_at(fit_in, contact["frame"], spec.fps, spec.surface)[1]
                    if fit_in
                    else None
                )
                v_out = fit_out.theta[3:6] if fit_out else None
                # Rescued one-sided skip-fit for a contact the adjacent-span logic abstained
                # on (an insertion split its only flight into a same-side stub). Never
                # overrides an existing flight — pure fallback for an otherwise-lost contact.
                rescued = rescue.get(i)
                if rescued is not None:
                    if v_in is None:
                        v_in = rescued["v_in"]
                    if v_out is None:
                        v_out = rescued["v_out"]
                racket = implied_racket(v_in, v_out)
                point = fused[i]
                if rescued is not None and "pos" in rescued and fit_in is None and fit_out is None:
                    point = rescued["pos"]
                # Interval-valued timing: half-width from the closest-approach curvature,
                # widened when the audio witness disagrees (calibrated coverage from BOTH the
                # image-distance well AND witness agreement, per the interval-timing spec).
                onset, a_delta, a_abstain = audio_witness_check(
                    contact["frame"], audio_onsets, witness
                )
                timing_ci = contact.get("timing_ci_frames", 2.0)
                if a_abstain and a_delta is not None:
                    timing_ci = min(5.0, max(timing_ci, abs(a_delta) / 2))
                frame_lo = contact["frame"] - timing_ci
                frame_hi = contact["frame"] + timing_ci
                gap = gaps2.get(i, 0.0)
                ci = 1.96 * (
                    0.1
                    + 0.15 * gap
                    + 0.035 * max(fit_in.rms_px if fit_in else 0, fit_out.rms_px if fit_out else 0)
                )
                supported = fit_in is not None or fit_out is not None or rescued is not None
                row = dict(
                    match=match,
                    clip=clip,
                    contact_index=i,
                    frame=f"{contact['frame']:.3f}",
                    frame_detector=f"{contact['frame_detector']:.3f}",
                    frame_delta=f"{contact['frame'] - contact['frame_detector']:.3f}",
                    frame_lo=f"{frame_lo:.3f}",
                    frame_hi=f"{frame_hi:.3f}",
                    t_lo_s=f"{frame_lo / spec.fps:.6f}",
                    t_hi_s=f"{frame_hi / spec.fps:.6f}",
                    timing_source=contact.get("retime_source", "detector"),
                    timing_residual_px=(
                        f"{contact['retime_residual_px']:.3f}"
                        if "retime_residual_px" in contact
                        else ""
                    ),
                    side=contact["side"],
                    phase=contact["phase"],
                    status="fit" if supported else "unsupported_dead_or_same_side",
                    x=f"{point[0]:.4f}",
                    y=f"{point[1]:.4f}",
                    z=f"{point[2]:.4f}",
                    ci95_x_m=f"{ci:.3f}",
                    ci95_y_m=f"{ci:.3f}",
                    ci95_z_m=f"{ci:.3f}",
                    junction_gap_pass1_m=f"{gaps1.get(i, math.nan):.4f}" if i in gaps1 else "",
                    junction_gap_pass2_m=f"{gaps2.get(i, math.nan):.4f}" if i in gaps2 else "",
                    rms_in_px=f"{fit_in.rms_px:.3f}" if fit_in else "",
                    rms_out_px=f"{fit_out.rms_px:.3f}" if fit_out else "",
                    velocity_ci95_ms=f"{0.5 + 0.12 * max(fit_in.rms_px if fit_in else 0, fit_out.rms_px if fit_out else 0):.3f}",
                    racket_normal_ci95_deg=f"{min(90.0, 8.0 + 5.0 * ci):.3f}",
                    racket_speed_ci95_ms=f"{1.0 + 0.25 * max(fit_in.rms_px if fit_in else 0, fit_out.rms_px if fit_out else 0):.3f}",
                    uncertainty_method="heuristic_fit_rms_plus_junction_gap",
                    timing_audio_onset=(f"{onset:.1f}" if onset is not None else ""),
                    timing_audio_delta_f=(f"{a_delta:.2f}" if a_delta is not None else ""),
                    timing_audio_abstain=("1" if a_abstain else "0"),
                )
                for label, value in (("in", v_in), ("out", v_out)):
                    row.update(
                        {
                            f"vx_{label}": "",
                            f"vy_{label}": "",
                            f"vz_{label}": "",
                            f"speed_{label}": "",
                        }
                    )
                    if value is not None:
                        row.update(
                            {
                                f"vx_{label}": f"{value[0]:.4f}",
                                f"vy_{label}": f"{value[1]:.4f}",
                                f"vz_{label}": f"{value[2]:.4f}",
                                f"speed_{label}": f"{np.linalg.norm(value):.4f}",
                            }
                        )
                keys = [
                    "racket_normal_x",
                    "racket_normal_y",
                    "racket_normal_z",
                    "racket_normal_speed_ms",
                    "racket_face_yaw_deg",
                    "racket_face_pitch_deg",
                ]
                row.update({key: "" for key in keys})
                if racket:
                    normal, speed, yaw, pitch = racket
                    row.update(
                        racket_normal_x=f"{normal[0]:.4f}",
                        racket_normal_y=f"{normal[1]:.4f}",
                        racket_normal_z=f"{normal[2]:.4f}",
                        racket_normal_speed_ms=f"{speed:.4f}",
                        racket_face_yaw_deg=f"{yaw:.3f}",
                        racket_face_pitch_deg=f"{pitch:.3f}",
                    )
                cw.writerow(row)
    return ball_path, contact_path, bounce_path


def write_contact_audit(
    spec: MatchSpec,
    clip: str,
    ball_path: str,
    contact_path: str,
    count: int = 12,
    tag: str = "rich_ball_physics_v1",
    camera_file: str = "camera_P_per_frame_v1.npz",
):
    """Twelve deterministic 3-frame filmstrips: raw observation vs rich 3D projection."""
    contacts = [row for row in read_rows_any(contact_path, clip) if row["status"] == "fit"]
    if not contacts:
        return None
    ball_rows = read_rows_any(ball_path, clip)
    states = {
        int(row["frame"]): np.array([float(row[k]) for k in ("x", "y", "z")]) for row in ball_rows
    }
    raw = load_ball(spec.ball, clip)
    camera = FrameCamera(spec.out, clip, camera_file)
    frames_root, image_size = highest_frames(spec)
    artifact_size = res.CANONICAL_SIZE
    chosen = np.linspace(0, len(contacts) - 1, min(count, len(contacts))).round().astype(int)
    strips = []
    for index in chosen:
        contact = contacts[int(index)]
        center = int(round(float(contact["frame"])))
        panels = []
        for frame in (center - 1, center, center + 1):
            path = os.path.join(frames_root, clip, f"f_{frame:04d}.jpg")
            image = cv2.imread(path)
            if image is None:
                image = np.zeros((image_size.height, image_size.width, 3), dtype=np.uint8)
            if frame in raw:
                uv = tuple(
                    np.round(res.scale_points(raw[frame], artifact_size, image_size)).astype(int)
                )
                radius = int(round(res.pixel_length(8, artifact_size, image_size)))
                cv2.circle(image, uv, radius, (0, 210, 255), 2, cv2.LINE_AA)
            if frame in states:
                uv_fit = res.scale_points(
                    project_one(camera.p_at(frame), states[frame]), artifact_size, image_size
                )
                marker_size = int(round(res.pixel_length(18, artifact_size, image_size)))
                cv2.drawMarker(
                    image,
                    tuple(np.round(uv_fit).astype(int)),
                    (40, 40, 240),
                    cv2.MARKER_CROSS,
                    marker_size,
                    2,
                )
            if frame == center:
                cv2.putText(
                    image,
                    f"{clip} c{contact['contact_index']} f={contact['frame']} "
                    f"d={contact['frame_delta']} z={float(contact['z']):.2f}m",
                    (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            panels.append(cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA))
        strips.append(np.hstack(panels))
    blank = np.zeros_like(strips[0])
    while len(strips) % 2:
        strips.append(blank)
    montage = np.vstack([np.hstack(strips[i : i + 2]) for i in range(0, len(strips), 2)])
    path = os.path.join(spec.output_dir, f"{tag}_{clip}_audit{len(chosen)}.jpg")
    cv2.imwrite(path, montage)
    res.write_coordinate_manifest(
        f"{path}.coordinates.json",
        image_size=image_size,
        artifact_size=artifact_size,
        source=frames_root,
        extra={"clip": clip, "audit": os.path.basename(path)},
        subnative_flagged=True,
        subnative_justification=(
            "legacy physics audit overlay; Resolution Contract migration inventory item 17"
        ),
    )
    return path


def read_rows_any(path: str, clip: str):
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"
    with opener(path, mode, newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["clip"] == clip]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", required=True, choices=["rg2025f", "uso2025f"])
    parser.add_argument("--clips", nargs="*", help="clip ids; default: camera-P coverage")
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--contact-geometry", help="optional anisotropic contact-volume CSV")
    parser.add_argument(
        "--camera-file",
        default="camera_P_per_frame_v1.npz",
        help="per-frame projection NPZ within the processed match directory",
    )
    parser.add_argument("--tag", default="rich_ball_physics_v1", help="versioned output prefix")
    parser.add_argument(
        "--ball-override", default="", help="ball observations CSV replacing spec.ball"
    )
    parser.add_argument(
        "--contacts-override",
        default="",
        help="features-style contacts CSV replacing spec.features (e.g. knot-solver discoveries)",
    )
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    spec = match_spec(args.match)
    stage = StageRun(spec.out, args.tag, args, seed=args.seed)
    p_data = np.load(os.path.join(spec.out, args.camera_file))
    clips = args.clips or sorted(set(p_data["clips"].astype(str).tolist()))
    results = {}
    summary = []
    for clip in clips:
        camera = FrameCamera(spec.out, clip, args.camera_file)
        if not camera.available():
            print(f"{clip}: skipped (no per-frame camera)")
            continue
        contacts = load_contacts(args.contacts_override or spec.features, clip)
        ball = load_ball(args.ball_override or spec.ball, clip)
        retime_contacts(contacts, ball)
        players = load_players(spec.players, clip, camera)
        bounce_anchors = load_bounce_anchors(spec.bounce_times, clip, ball, camera)
        contact_geometry = load_contact_geometry(args.contact_geometry, clip)
        first = {}
        for i in range(len(contacts) - 1):
            s0, s1 = contacts[i]["side"], contacts[i + 1]["side"]
            interior = 0 < i and i + 1 < len(contacts) - 1
            if s0 == "unknown" and s1 == "unknown" and interior:
                # BOTH endpoints abstained on side. Baseline refused this span (equal
                # labels), so the contact abstained even when it was a real rally hit whose
                # motion-based side estimate simply failed. A live rally flight crosses the
                # net, so re-fit under both opposite-side player anchors and keep the
                # better-scoring geometry — the anchor penalty in the score rejects the
                # orientation that only fits pixels by ghosting the ball's depth. Confident
                # same-side spans (far-far / near-near: pre-serve bounces) stay refused
                # below, and spans with one known side keep their baseline fit untouched, so
                # this only ADDS coverage where the pipeline previously had none. The
                # interior guard excludes the clip's first/last contact: a genuinely rally-
                # opening or -closing hit has a known side (the serve, a clear groundstroke),
                # so a double-unknown at a clip boundary is the pre-serve toss / post-point
                # dead ball, not a shot — admitting those was pure false-positive cost.
                fit = None
                for override in (("near", "far"), ("far", "near")):
                    cand = fit_shot(
                        i,
                        contacts,
                        ball,
                        players,
                        camera,
                        spec.fps,
                        spec.surface,
                        args.max_nfev,
                        side_override=override,
                    )
                    if cand is not None and (fit is None or cand._score < fit._score):
                        fit = cand
                # Energy gate: admit only if the flight is fast enough to be a struck ball,
                # so a low-energy roll/bounce between two unknowns stays abstained.
                if fit is not None and float(np.linalg.norm(fit.theta[3:6])) < AMBIG_SPEED_MIN_MS:
                    fit = None
            elif s0 == s1:
                # Confident same known side (far-far / near-near), OR a boundary/low-value
                # double-unknown that fell through the interior guard: not a live cross-net
                # flight (pre-serve bounce, dead ball, or a same-side pair hiding a contact).
                # Refuse, as baseline did (fit_shot no longer self-rejects equal sides).
                fit = None
            else:
                # Opposite known sides, or exactly one unknown side: baseline behaviour.
                fit = fit_shot(
                    i, contacts, ball, players, camera, spec.fps, spec.surface, args.max_nfev
                )
            if fit is not None:
                first[i] = fit
        # Rescue contacts left with no fittable flight because a same-side / too-close
        # insertion split their only cross-net flight into an unfittable stub. Runs AFTER
        # the adjacent-span logic so it only ever adds fits, never perturbs existing ones.
        # RBP_NO_RESCUE=1 is a kill-switch / ablation control (pre-rescue behaviour).
        if os.environ.get("RBP_NO_RESCUE") == "1":
            rescue = {}
        else:
            rescue = rescue_abstaining_contacts(
                contacts, first, ball, players, camera, spec.fps, spec.surface, args.max_nfev
            )
        fused1, gaps1 = fuse_contacts(
            contacts, first, players, ball, camera, spec.fps, spec.surface
        )
        initial_positions = dict(fused1)
        for contact_id, geometry in contact_geometry.items():
            if contact_id in initial_positions:
                initial_positions[contact_id] = geometry["center"].copy()
        _joint_positions, joint_fits = jointly_refine_all(
            contacts,
            ball,
            players,
            camera,
            spec.fps,
            spec.surface,
            first,
            initial_positions,
            args.max_nfev,
            bounce_anchors,
            contact_geometry,
        )
        # The sparse joint solution is the published pass. Exact boundary shooting is a
        # diagnostic only: the first campaign showed it can hide an endpoint error by
        # increasing image residual 2-4x. Shared variables + observed bounce anchors retain
        # the honest compromise and expose any remaining junction gap.
        second = joint_fits
        # Advisor Q4: retime interior junctions to the closest approach of the refined
        # incoming/outgoing flights in image space (sub-frame). Mutates contact["frame"], so
        # it runs BEFORE fuse_contacts (v_in is then read at the corrected contact time).
        retime_closest_approach(contacts, second, camera, spec.fps, spec.surface)
        fused2, gaps2 = fuse_contacts(
            contacts, second, players, ball, camera, spec.fps, spec.surface
        )
        results[clip] = contacts, second, fused2, gaps1, gaps2, rescue
        supported_z = [fused2[i][2] for i in range(len(contacts)) if i in second or i - 1 in second]
        bad_z = sum(not (0.2 <= z <= 3.5) for z in supported_z)
        values1, values2 = list(gaps1.values()), list(gaps2.values())
        row = {
            "clip": clip,
            "contacts": len(contacts),
            "shots_fit": len(second),
            "obs": sum(f.n_obs for f in second.values()),
            "bad_contact_z": bad_z,
            "junction_median_pass1": float(np.median(values1)) if values1 else math.nan,
            "junction_max_pass1": max(values1, default=math.nan),
            "junction_median_pass2": float(np.median(values2)) if values2 else math.nan,
            "junction_max_pass2": max(values2, default=math.nan),
            "rms_median_px": float(np.median([f.rms_px for f in second.values()]))
            if second
            else math.nan,
            "rms_max_px": max((f.rms_px for f in second.values()), default=math.nan),
        }
        bounces = [bounce for fit in second.values() for bounce in fit.bounces]
        row["bounce_count"] = len(bounces)
        row["bounce_in_court_rate"] = (
            sum(0 <= b["x"][0] <= 10.97 and 0 <= b["x"][1] <= 23.77 for b in bounces) / len(bounces)
            if bounces
            else 0.0
        )
        row["accepted"] = bool(
            bad_z <= 2
            and row["junction_median_pass2"] <= 0.5
            and row["junction_max_pass2"] <= 1.5
            and row["rms_median_px"] <= 3.0
            and row["bounce_in_court_rate"] >= 0.9
        )
        summary.append(row)
        print(
            f"{clip}: {row['shots_fit']} shots, {row['obs']} obs | z_bad={bad_z}/{len(supported_z)} "
            f"| junction median {row['junction_median_pass1']:.2f}->{row['junction_median_pass2']:.2f}m "
            f"max {row['junction_max_pass1']:.2f}->{row['junction_max_pass2']:.2f}m "
            f"| RMS median/max {row['rms_median_px']:.2f}/{row['rms_max_px']:.2f}px"
        )
    outputs = write_outputs(spec, args.match, clips, results, args.tag)
    audits = [
        write_contact_audit(
            spec, clip, outputs[0], outputs[1], tag=args.tag, camera_file=args.camera_file
        )
        for clip in clips
    ]
    quality_path = os.path.join(spec.output_dir, f"{args.tag}_quality.json")
    with open(quality_path, "w") as handle:
        json.dump(
            {
                "version": args.tag,
                "accepted": bool(summary) and all(row["accepted"] for row in summary),
                "gate": {
                    "bad_contact_z_max": 2,
                    "junction_median_m_max": 0.5,
                    "junction_max_m_max": 1.5,
                    "rms_median_px_max": 3.0,
                    "bounce_in_court_rate_min": 0.9,
                },
                "clips": summary,
                "tier1_labels_consumed": False,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    stage.finish(
        outputs={
            "clips": summary,
            "ball": outputs[0],
            "contacts": outputs[1],
            "bounces": outputs[2],
            "audits": [path for path in audits if path],
            "quality": quality_path,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
