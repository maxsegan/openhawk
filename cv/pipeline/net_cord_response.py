"""Net-cord response as an admissible set, not a VR tape clip.

Owner-directed correctness change, 2026-09-20. Default off: absent and
``tape_clip`` are the existing ``net_impact_velocity`` law. ``admissible_set``
means a declared or proposed net hit is admissible if there exists a post-contact
velocity in the box that matches the observed pixels. ``evidence_bound`` does
not search that set: it clips the law's outgoing velocity into the pixel
ellipsoid the outgoing evidence already supports, and abstains to the tape
clip when that evidence is missing or ambiguous. Extra accepts on
no-net-contact flights are out of scope and are the unrecoverable error.
"""

from __future__ import annotations

import math
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from cv.pipeline.physics_knot_solver import NET_Y, net_impact_velocity, net_tape_height
from cv.pipeline.rich_ball_physics import R_BALL
from physics import flight as aero

FIELD = "net_cord_response"
TAPE_BAND_FIELD = "net_cord_tape_band_m"
TAPE_CLIP = "tape_clip"
ADMISSIBLE_SET = "admissible_set"
EVIDENCE_BOUND = "evidence_bound"
MODES = (TAPE_CLIP, ADMISSIBLE_SET, EVIDENCE_BOUND)
RECEIPT_SCHEMA = "net_cord_response_v1"

R_MIN = 0.03
R_MAX = 0.95
H_TOL_M = 0.05
H_TOL_WIDE_M = 0.20
CONE_HALF_ANGLE_DEG = 90.0
INPLANE_FLOOR_MPS = 0.5
POST_CONTACT_WIDENING = 3.0
PROMOTED_RMS_LIMIT_PX = 16.0
WIDENED_RMS_LIMIT_PX = PROMOTED_RMS_LIMIT_PX * POST_CONTACT_WIDENING
# Evidence-bound uses this as the pixel-set radius, not as an objective weight.
PIXEL_TOL_PX = WIDENED_RMS_LIMIT_PX

_MODE: ContextVar[str] = ContextVar("net_cord_response_mode", default=TAPE_CLIP)
_H_TOL: ContextVar[float] = ContextVar("net_cord_tape_band_m", default=H_TOL_M)
_BOUNDS: ContextVar[tuple] = ContextVar("net_cord_evidence_bounds", default=())


MID_FLIGHT_FIELD = "net_cord_mid_flight"
MID_FLIGHT_MODES = ("off", ADMISSIBLE_SET)


def mid_flight_net_hits(events: list[dict]) -> list[float]:
    """Labelled net hits the rally carries on: next a bounce, then a later contact.

    ``net_cord_mid_flight="admissible_set"`` gives only such an attempt or
    component the admissible response. A net stop, a net-then-ground ending or a
    serve that ends at its bounce is not mid-flight and keeps the tape clip.
    """

    ordered = sorted(
        (e for e in events if e["event_type"] in {"contact", "bounce", "net_hit"}),
        key=lambda e: float(e["frame"]),
    )
    found = []
    for index, event in enumerate(ordered):
        if event["event_type"] != "net_hit":
            continue
        later = ordered[index + 1 :]
        if not any(e["event_type"] == "contact" for e in ordered[:index]):
            continue
        if not later or later[0]["event_type"] != "bounce":
            continue
        if any(e["event_type"] == "contact" for e in later[1:]):
            found.append(float(event["frame"]))
    return found


def validate_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(
            "net_cord_response must be 'tape_clip', 'admissible_set', or 'evidence_bound'"
        )
    return mode


def validate_h_tol(value) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ValueError("net_cord_tape_band_m must be a non-negative finite length in metres")
    width = float(value)
    if not math.isfinite(width) or width < 0.0:
        raise ValueError("net_cord_tape_band_m must be a non-negative finite length in metres")
    return width


def active_mode() -> str:
    return _MODE.get()


def active_h_tol() -> float:
    return _H_TOL.get()


class using_mode:
    """Process-local mode and tape-band width. Re-enterable: refine then measure."""

    def __init__(self, mode: str, h_tol: float | None = None):
        self._mode = mode
        self._h_tol = h_tol

    def __enter__(self):
        self._mode_token = _MODE.set(validate_mode(self._mode))
        self._h_token = None if self._h_tol is None else _H_TOL.set(validate_h_tol(self._h_tol))
        return self

    def __exit__(self, *exc):
        _MODE.reset(self._mode_token)
        if self._h_token is not None:
            _H_TOL.reset(self._h_token)
        return False


def tape_band(x: float, h_tol: float | None = None) -> tuple[float, float]:
    tape = float(net_tape_height(x))
    width = active_h_tol() if h_tol is None else float(h_tol)
    return tape - R_BALL - width, tape + R_BALL + width


def in_tape_band(xyz, h_tol: float | None = None) -> bool:
    values = np.asarray(xyz, float)
    if values.shape != (3,) or not np.isfinite(values).all():
        return False
    lo, hi = tape_band(float(values[0]), h_tol)
    return lo <= float(values[2]) <= hi


def contact_covariates(xyz) -> dict:
    values = np.asarray(xyz, float)
    tape = float(net_tape_height(float(values[0])))
    return {
        "x_on_tape": float(values[0]),
        "height_offset": float(values[2]) - tape,
        "tape_height_m": tape,
        "net_y_m": NET_Y,
    }


@dataclass(frozen=True)
class EvidenceBound:
    """Pixel ellipsoid of outgoing velocities. Abstain leaves the tape clip alone.

    ``axes`` is Vh from the pixel-Jacobian SVD. Admitted velocities satisfy
    ``||S Vh (v - v_center)||^2 <= tau``.
    """

    net_frame: float
    admitted: bool
    reason: str
    h_tol_m: float
    observation_count: int = 0
    sources: tuple = ()
    pixel_rms_px: float | None = None
    v_center: tuple | None = None
    singular_values: tuple | None = None
    axes: tuple | None = None
    tau: float | None = None

    def to_dict(self) -> dict:
        return {
            "net_frame": self.net_frame,
            "admitted": self.admitted,
            "reason": self.reason,
            "h_tol_m": self.h_tol_m,
            "observation_count": self.observation_count,
            "sources": list(self.sources),
            "pixel_rms_px": self.pixel_rms_px,
            "v_center_mps": None if self.v_center is None else list(self.v_center),
            "singular_values": None if self.singular_values is None else list(self.singular_values),
            "axes": None if self.axes is None else [list(row) for row in self.axes],
            "tau": self.tau,
            FIELD: EVIDENCE_BOUND,
            "searches_v_out": False,
        }


def bound_from_record(record: dict) -> EvidenceBound:
    center = record.get("v_center_mps") if "v_center_mps" in record else record.get("v_center")
    pixel_set = record.get("pixel_set") or {}
    if center is None:
        center = pixel_set.get("v_center_mps")
    axes = record.get("axes") or pixel_set.get("axes")
    singular = record.get("singular_values") or pixel_set.get("singular_values")
    tau = record.get("tau") if record.get("tau") is not None else pixel_set.get("tau")
    return EvidenceBound(
        net_frame=float(record["net_frame"]),
        admitted=bool(record.get("admitted")),
        reason=str(record.get("reason") or record.get("abstain_reason") or ""),
        h_tol_m=float(record.get("h_tol_m", active_h_tol())),
        observation_count=int(record.get("observation_count") or 0),
        sources=tuple(record.get("sources") or ()),
        pixel_rms_px=record.get("pixel_rms_px"),
        v_center=None if center is None else tuple(float(v) for v in center),
        singular_values=None if singular is None else tuple(float(v) for v in singular),
        axes=None if axes is None else tuple(tuple(float(v) for v in row) for row in axes),
        tau=None if tau is None else float(tau),
    )


def bounds_from_records(records) -> tuple:
    return tuple(bound_from_record(row) for row in records or ())


class using_evidence:
    """Re-enterable evidence-bound mode. Abstaining bounds clip nothing."""

    def __init__(self, bounds, h_tol: float | None = None):
        self._bounds = tuple(bounds)
        self._h_tol = h_tol

    def __enter__(self):
        self._mode_token = _MODE.set(EVIDENCE_BOUND)
        self._h_token = None if self._h_tol is None else _H_TOL.set(validate_h_tol(self._h_tol))
        self._b_token = _BOUNDS.set(self._bounds)
        return self

    def __exit__(self, *exc):
        _MODE.reset(self._mode_token)
        if self._h_token is not None:
            _H_TOL.reset(self._h_token)
        _BOUNDS.reset(self._b_token)
        return False


def active_bounds() -> tuple:
    return _BOUNDS.get()


def matching_bound(net_frame: float):
    frame = float(net_frame)
    best = None
    best_gap = None
    for bound in _BOUNDS.get():
        gap = abs(float(bound.net_frame) - frame)
        if gap <= 1.5 and (best_gap is None or gap < best_gap):
            best, best_gap = bound, gap
    return best


def uses_tape_band(net_frame=None) -> bool:
    """Tape band for the free search, or for one admitted evidence bound."""
    if active_mode() == ADMISSIBLE_SET:
        return True
    if active_mode() != EVIDENCE_BOUND:
        return False
    if net_frame is None:
        return any(bound.admitted for bound in _BOUNDS.get())
    bound = matching_bound(net_frame)
    return bool(bound is not None and bound.admitted)


def pixel_metric(bound: EvidenceBound, velocity) -> float:
    if not bound.admitted or bound.v_center is None or bound.axes is None:
        return float("inf")
    dv = np.asarray(velocity, float) - np.asarray(bound.v_center, float)
    coords = np.asarray(bound.axes, float) @ dv
    singular = np.asarray(bound.singular_values, float)
    return float(np.sum((singular * coords) ** 2))


def in_pixel_set(bound: EvidenceBound, velocity) -> bool:
    if bound.tau is None:
        return False
    return pixel_metric(bound, velocity) <= float(bound.tau) * (1.0 + 1e-6)


def project_pixel_set(bound: EvidenceBound, velocity) -> np.ndarray:
    """Closed-form clip onto the pixel ellipsoid. Not a search."""
    velocity = np.asarray(velocity, float)
    if not bound.admitted or bound.v_center is None or bound.tau is None or bound.axes is None:
        return velocity.copy()
    center = np.asarray(bound.v_center, float)
    axes = np.asarray(bound.axes, float)
    coords = axes @ (velocity - center)
    singular = np.asarray(bound.singular_values, float)
    metric = float(np.sum((singular * coords) ** 2))
    if metric <= float(bound.tau) or metric <= 0.0:
        return velocity.copy()
    return center + axes.T @ (coords * math.sqrt(float(bound.tau) / metric))


def constrain_outgoing(v_in, v_law, net_frame) -> np.ndarray:
    """Clip the law into the pixel set and the physics box.

    A velocity already inside both is unchanged, so the set is not collapsed
    to one fitted point. Absent or ambiguous evidence returns the law.
    """
    law = np.asarray(v_law, float)
    if active_mode() != EVIDENCE_BOUND:
        return law
    bound = matching_bound(net_frame)
    if bound is None or not bound.admitted:
        return law
    boxed = project_to_box(v_in, project_pixel_set(bound, law))
    if in_pixel_set(bound, boxed) and in_box(v_in, boxed):
        return boxed
    return project_to_box(v_in, np.asarray(bound.v_center, float))


def evidence_witness(bound: EvidenceBound, hit: dict | None = None) -> dict:
    witness = receipt(EVIDENCE_BOUND, h_tol_m=bound.h_tol_m)
    witness.update(bound.to_dict())
    witness["admitted"] = bool(bound.admitted)
    witness["abstain_reason"] = None if bound.admitted else bound.reason
    witness["searches_v_out"] = False
    if hit is not None:
        outgoing = np.asarray(hit["v_out"], float)
        witness["v_out_mps"] = outgoing.tolist()
        witness["outgoing_velocity_mps"] = outgoing.tolist()
        witness["v_in_mps"] = np.asarray(hit.get("v_in", [0.0, 0.0, 0.0]), float).tolist()
        witness["in_tape_band"] = bool(in_tape_band(hit["x"], bound.h_tol_m))
        witness["covariates"] = contact_covariates(hit["x"])
        witness["model"] = "evidence_bound_net_response_v1"
    return witness


def adjust_incoming_to_tape(xyz, h_tol: float | None = None):
    """Snap z onto the tape when the cached crossing is already in the band.

    Cheap existence-search adjustment: 9–16 cm of incoming noise should not
    decide admissibility. Outside the band the origin is unchanged. The
    delta is a covariate, never a constraint.
    """
    origin = np.asarray(xyz, float).copy()
    covariates = contact_covariates(origin)
    width = active_h_tol() if h_tol is None else float(h_tol)
    covariates["h_tol_m"] = width
    covariates["incoming_z_adjustment_m"] = 0.0
    covariates["incoming_adjusted"] = False
    covariates["adjusted_z_m"] = float(origin[2]) if origin.shape == (3,) else None
    if origin.shape != (3,) or not np.isfinite(origin).all():
        return origin, covariates
    if not in_tape_band(origin, width):
        return origin, covariates
    tape = float(covariates["tape_height_m"])
    delta = tape - float(origin[2])
    origin[2] = tape
    covariates["incoming_z_adjustment_m"] = float(delta)
    covariates["incoming_adjusted"] = bool(abs(delta) > 1e-12)
    covariates["adjusted_z_m"] = float(origin[2])
    return origin, covariates


def tape_clip_seed(v_in) -> np.ndarray:
    return np.asarray(net_impact_velocity(np.asarray(v_in, float)), float)


def _inplane(v) -> np.ndarray:
    return np.array([float(v[0]), float(v[2])], float)


def in_plane_cone(v_in, v_out, half_angle_deg: float = CONE_HALF_ANGLE_DEG) -> bool:
    incoming = _inplane(v_in)
    outgoing = _inplane(v_out)
    speed_in = float(np.linalg.norm(incoming))
    speed_out = float(np.linalg.norm(outgoing))
    if speed_in < INPLANE_FLOOR_MPS or speed_out < 1e-12:
        return True
    cosine = float(np.clip(incoming @ outgoing / (speed_in * speed_out), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine))) <= half_angle_deg + 1e-9


def speed_ratio(v_in, v_out) -> float:
    incoming = float(np.linalg.norm(v_in))
    outgoing = float(np.linalg.norm(v_out))
    if incoming < 1e-12:
        return 0.0 if outgoing < 1e-12 else float("inf")
    return outgoing / incoming


def in_box(v_in, v_out) -> bool:
    incoming = np.asarray(v_in, float)
    outgoing = np.asarray(v_out, float)
    if incoming.shape != (3,) or outgoing.shape != (3,):
        return False
    if not np.isfinite(incoming).all() or not np.isfinite(outgoing).all():
        return False
    ratio = speed_ratio(incoming, outgoing)
    if not (R_MIN - 1e-12 <= ratio <= R_MAX + 1e-12):
        return False
    if abs(float(outgoing[1])) > abs(float(incoming[1])) + 1e-9:
        return False
    return in_plane_cone(incoming, outgoing)


def project_to_box(v_in, v_out) -> np.ndarray:
    """Nearest point in the admissible box; used as a search constraint, not a law."""
    incoming = np.asarray(v_in, float)
    outgoing = np.asarray(v_out, float).copy()
    speed_in = float(np.linalg.norm(incoming))
    if speed_in < 1e-12:
        return np.zeros(3)
    limit = abs(float(incoming[1]))
    outgoing[1] = float(np.clip(outgoing[1], -limit, limit))
    plane_in = _inplane(incoming)
    plane_out = _inplane(outgoing)
    in_speed = float(np.linalg.norm(plane_in))
    if in_speed >= INPLANE_FLOOR_MPS:
        projection = float(plane_in @ plane_out)
        if projection < 0.0:
            plane_out = plane_out - (projection / (in_speed * in_speed)) * plane_in
            outgoing[0], outgoing[2] = float(plane_out[0]), float(plane_out[1])
    speed_out = float(np.linalg.norm(outgoing))
    if speed_out < 1e-12:
        outgoing = tape_clip_seed(incoming)
        speed_out = float(np.linalg.norm(outgoing))
    ratio = speed_out / speed_in
    if ratio < R_MIN:
        outgoing = outgoing * (R_MIN * speed_in / max(speed_out, 1e-12))
    elif ratio > R_MAX:
        outgoing = outgoing * (R_MAX * speed_in / speed_out)
    return outgoing


def observation_seed(x0, t0, samples, fps: float) -> np.ndarray | None:
    """Ballistic seed from the first post-contact 3D sample; None if unusable."""
    if not samples:
        return None
    frame, xyz = samples[0]
    dt = (float(frame) - float(t0)) / float(fps)
    if not np.isfinite(dt) or dt <= 1e-4:
        return None
    delta = np.asarray(xyz, float) - np.asarray(x0, float)
    if delta.shape != (3,) or not np.isfinite(delta).all():
        return None
    velocity = delta / dt
    velocity[2] += 0.5 * aero.default_params["g"] * dt
    return velocity


def _propagate(x0, v_out, spin, times_s):
    positions, _, _ = aero.sample_states(
        np.asarray(x0, float),
        np.asarray(v_out, float),
        np.asarray(spin, float),
        np.asarray(times_s, float),
    )
    return positions


@dataclass(frozen=True)
class ExistenceResult:
    admissible: bool
    v_out: np.ndarray
    residual_rms: float
    raw_residual_rms: float
    seed: str
    in_box: bool
    nfev: int
    covariates: dict


def search_admissible_v_out(
    v_in,
    x0,
    t0: float,
    samples: list[tuple[float, np.ndarray]],
    fps: float,
    spin=None,
    *,
    cameras=None,
    pixels=None,
    radial=None,
    widening: float = POST_CONTACT_WIDENING,
    h_tol: float | None = None,
) -> ExistenceResult:
    """Box-constrained existence search for post-contact ``v_out``.

    ``samples`` are bounce-free post-contact (frame, xyz_m) targets used when
    cameras are absent. With cameras, the residual is native pixels. Post-contact
    residuals are divided by ``widening`` (3× vs the 16 px promoted gate).
    Incoming contact inside the tape band is snapped onto the tape; the
    adjustment is recorded, not constrained.
    """
    incoming = np.asarray(v_in, float)
    origin, covariates = adjust_incoming_to_tape(x0, h_tol)
    spin_vector = np.zeros(3) if spin is None else np.asarray(spin, float)
    clip = project_to_box(incoming, tape_clip_seed(incoming))
    observed = observation_seed(origin, t0, samples, fps)
    seeds = [("tape_clip", clip)]
    if observed is not None:
        seeds.append(("observations", project_to_box(incoming, observed)))
    drop = project_to_box(incoming, np.array([incoming[0], -0.1 * incoming[1], incoming[2]]))
    seeds.append(("near_dead_drop", drop))

    frames = np.asarray([s[0] for s in samples], float)
    times = (frames - float(t0)) / float(fps)
    use_pixels = cameras is not None and pixels is not None
    if use_pixels:
        from cv.experiments.connected_shooting.camera_geometry import project

        targets = np.asarray(pixels, float)
    else:
        targets = np.asarray([s[1] for s in samples], float)

    def residual(velocity):
        outgoing = project_to_box(incoming, velocity)
        predicted = _propagate(origin, outgoing, spin_vector, times)
        if use_pixels:
            predicted = project(np.asarray(cameras, float), predicted, radial)
        error = (predicted - targets).ravel()
        return error / float(widening)

    best = None
    for name, seed in seeds:
        try:
            fit = least_squares(
                residual,
                np.asarray(seed, float),
                method="trf",
                max_nfev=40,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        candidate = project_to_box(incoming, fit.x)
        values = residual(candidate)
        raw = values * float(widening)
        rms = float(np.sqrt(np.mean(values**2))) if len(values) else float("inf")
        raw_rms = float(np.sqrt(np.mean(raw**2))) if len(raw) else float("inf")
        row = (rms, name, candidate, raw_rms, int(fit.nfev), bool(in_box(incoming, candidate)))
        if best is None or row[0] < best[0]:
            best = row
    if best is None:
        return ExistenceResult(
            False,
            clip,
            float("inf"),
            float("inf"),
            "tape_clip",
            in_box(incoming, clip),
            0,
            covariates,
        )
    rms, name, velocity, raw_rms, nfev, boxed = best
    return ExistenceResult(
        admissible=bool(boxed and np.isfinite(rms) and raw_rms <= WIDENED_RMS_LIMIT_PX),
        v_out=velocity,
        residual_rms=rms,
        raw_residual_rms=raw_rms,
        seed=name,
        in_box=boxed,
        nfev=nfev,
        covariates=covariates,
    )


def receipt(mode: str, **fields) -> dict:
    validate_mode(mode)
    if mode == TAPE_CLIP:
        return {}
    if mode == EVIDENCE_BOUND:
        return {
            "schema": RECEIPT_SCHEMA,
            FIELD: mode,
            "constraint": "pixel_ellipsoid_intersect_physics_box",
            "searches_v_out": False,
            "abstain": "absent_or_ambiguous_outgoing_evidence_uses_tape_clip",
            "pixel_tolerance_px": PIXEL_TOL_PX,
            "speed_ratio_bounds": [R_MIN, R_MAX],
            "h_tol_m": active_h_tol(),
            "cone_half_angle_deg": CONE_HALF_ANGLE_DEG,
            **fields,
        }
    return {
        "schema": RECEIPT_SCHEMA,
        FIELD: mode,
        "speed_ratio_bounds": [R_MIN, R_MAX],
        "h_tol_m": active_h_tol(),
        "cone_half_angle_deg": CONE_HALF_ANGLE_DEG,
        "post_contact_residual_widening": POST_CONTACT_WIDENING,
        "widened_rms_limit_px": WIDENED_RMS_LIMIT_PX,
        "promoted_rms_limit_px": PROMOTED_RMS_LIMIT_PX,
        **fields,
    }


def flight_witness(hit: dict, result: ExistenceResult, mode: str) -> dict:
    """Self-describing row the verdict and the replay both read."""
    xyz = np.asarray(
        hit.get("x", result.covariates and [result.covariates["x_on_tape"], NET_Y, 0.0])
    )
    width = result.covariates.get("h_tol_m", active_h_tol())
    return receipt(
        mode,
        h_tol_m=width,
        v_out_mps=np.asarray(result.v_out, float).tolist(),
        v_in_mps=np.asarray(hit.get("v_in", [0.0, 0.0, 0.0]), float).tolist(),
        admissible=bool(result.admissible),
        in_tape_band=in_tape_band(xyz, width),
        in_box=bool(result.in_box),
        residual_rms_widened=result.residual_rms,
        residual_rms_raw_px=result.raw_residual_rms,
        seed=result.seed,
        covariates=result.covariates,
        spin_law="incoming_preserved_unconstrained",
    )
