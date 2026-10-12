"""Opt-in fitting witness for native nominal centers; never an acceptance witness.

At each supplied impact epoch, independent gravity wings share one ground point.
Native projection matrices preserve PTZ and camera translation. Uncertainty is
a statistical block plus conservative epoch/window/model discrepancy floors,
not a calibrated confidence region. No fit output or acceptance score is read.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from cv.experiments.connected_shooting import model

POLICY = dict(
    schema="fit_ground_witness_center_v1",
    physical_bound_policy="flag_with_heuristic_weight",
    wing_samples=4,
    auxiliary_wing_samples=[3, 6],
    auxiliary_availability="original_row_counts_before_fitting",
    maximum_distance_frames=12.0,
    profile_steps=101,
    pixel_sigma_floor=2.0,
    gravity_mps2=9.81,
    ball_radius_m=0.0325,
    maximum_irls_iterations=12,
    irls_relative_tolerance=1e-6,
    normalized_condition_limit=1e10,
    fixed_camera_center_tolerance_m=1e-6,
)


def _epochs(interval):
    lo, hi = interval
    return np.array([lo]) if lo == hi else np.linspace(lo, hi, POLICY["profile_steps"])


def _condition(a):
    scale = np.linalg.norm(a, axis=0)
    if np.any(scale <= 0):
        return float("inf")
    return float(np.linalg.cond(a / scale))


def _solve(a, b, weights, dt, incoming):
    matrix, rhs = a * weights[:, None], b * weights
    condition = _condition(matrix)
    if not np.isfinite(condition) or condition > POLICY["normalized_condition_limit"]:
        raise ValueError("rank_deficient_or_ill_conditioned")
    # Every negative dt requires vz <= g*dt/2; their intersection uses MIN.
    bounds = {4: np.min(4.905 * dt[incoming]), 7: np.max(4.905 * dt[~incoming])}
    candidates = []
    for active in ((), (4,), (7,), (4, 7)):
        z = np.zeros(8)
        free = [i for i in range(8) if i not in active]
        for index in active:
            z[index] = bounds[index]
        values, _, rank, _ = np.linalg.lstsq(matrix[:, free], rhs - matrix @ z, rcond=None)
        if rank != len(free):
            continue
        z[free] = values
        if z[4] > bounds[4] + 1e-9 or z[7] < bounds[7] - 1e-9:
            continue
        candidates.append((float(np.sum((matrix @ z - rhs) ** 2)), z, active))
    if not candidates:
        raise ValueError("rank_deficient_or_ill_conditioned")
    _, z, active = min(candidates, key=lambda row: row[0])
    return z, active, condition


def _project(z, dt, incoming, cameras):
    xyz = np.r_[z[:2], POLICY["ball_radius_m"]] + dt[:, None] * np.where(
        incoming[:, None], z[2:5], z[5:8]
    )
    xyz[:, 2] -= 4.905 * dt**2
    homogeneous = np.einsum("nij,nj->ni", cameras, np.c_[xyz, np.ones(len(xyz))])
    return homogeneous[:, :2] / homogeneous[:, 2:], homogeneous[:, 2], xyz


def _profile(frames, pixels, sigmas, cameras, interval, fps, wing):
    lo, hi = interval
    before = np.flatnonzero(frames < lo)[-wing:]
    after = np.flatnonzero(frames > hi)[:wing]
    if min(len(before), len(after)) < wing:
        raise ValueError("insufficient_source_support")
    ids = np.r_[before, after]
    f, uv, sigma, p = frames[ids], pixels[ids], sigmas[ids], cameras[ids]
    incoming = np.arange(len(ids)) < len(before)
    rows = []
    for epoch in _epochs(interval):
        dt = (f - epoch) / fps
        planes = p[:, :2] - uv[:, :, None] * p[:, 2:3]
        a = np.zeros((len(f), 2, 8))
        a[:, :, :2] = planes[:, :, :2]
        a[incoming, :, 2:5] = planes[incoming, :, :3] * dt[incoming, None, None]
        a[~incoming, :, 5:8] = planes[~incoming, :, :3] * dt[~incoming, None, None]
        b = -planes[:, :, 2] * (POLICY["ball_radius_m"] - 4.905 * dt**2)[:, None] - planes[:, :, 3]
        a, b = a.reshape(-1, 8), b.ravel()
        weights = 1 / np.maximum(np.linalg.norm(planes[:, :, :3], axis=2).ravel(), 1e-12)
        for iteration in range(POLICY["maximum_irls_iterations"]):
            z, active, condition = _solve(a, b, weights, dt, incoming)
            pred, depth, xyz = _project(z, dt, incoming, p)
            if np.any(np.abs(depth) < 1e-10) or not np.isfinite(pred).all():
                raise ValueError("nonfinite_projection")
            updated = np.repeat(1 / (sigma * np.abs(depth)), 2)
            relative_change = float(np.max(np.abs(updated / weights - 1)))
            weights = updated
            if relative_change <= POLICY["irls_relative_tolerance"]:
                break
        residual = (pred - uv) / sigma[:, None]
        rows.append(
            dict(
                epoch=float(epoch),
                xyz_m=np.r_[z[:2], POLICY["ball_radius_m"]].tolist(),
                parameters=z.tolist(),
                chi2=float(np.sum(residual**2)),
                rms_px=float(np.sqrt(np.mean((pred - uv) ** 2))),
                active_ground_bounds=list(active),
                normalized_condition=condition,
                irls_relative_change=relative_change,
                irls_iterations=iteration + 1,
                minimum_sample_height_m=float(np.min(xyz[:, 2])),
            )
        )
    best = min(rows, key=lambda row: row["chi2"])
    return dict(best=best, profile=rows, ids=ids, frames=f.tolist())


def _physical_flag(profile, interval):
    best = profile["best"]
    if best["active_ground_bounds"]:
        return "active_ground_bound"
    if best["minimum_sample_height_m"] < POLICY["ball_radius_m"] - 1e-9:
        return "subground_wing"
    if interval[0] != interval[1] and best["epoch"] in interval:
        return "epoch_at_interval_boundary"
    return None


def _quadratic_ground(frames, pixels, sigmas, cameras, interval):
    """Competing curvature estimate only for a genuinely fixed camera center."""
    ref = cameras[0]
    try:
        centers = np.array([-np.linalg.solve(p[:, :3], p[:, 3]) for p in cameras])
        h = np.array([ref[:, :3] @ np.linalg.inv(p[:, :3]) for p in cameras])
    except np.linalg.LinAlgError:
        return None, "singular_camera"
    if (
        np.max(np.linalg.norm(centers - centers[0], axis=1))
        > POLICY["fixed_camera_center_tolerance_m"]
    ):
        return None, "translated_camera"
    projected = np.einsum("nij,nj->ni", h, np.c_[pixels, np.ones(len(pixels))])
    uv = projected[:, :2] / projected[:, 2:]
    incoming = frames < interval[0]
    rows = []
    for epoch in _epochs(interval):
        dt = frames - epoch
        a = np.zeros((len(dt), 5))
        a[:, 0] = 1
        a[incoming, 1:3] = np.c_[dt[incoming], dt[incoming] ** 2]
        a[~incoming, 3:5] = np.c_[dt[~incoming], dt[~incoming] ** 2]
        beta, _, rank, _ = np.linalg.lstsq(a / sigmas[:, None], uv / sigmas[:, None], rcond=None)
        if rank != 5:
            return None, "rank_deficient"
        rows.append((float(np.sum(((a @ beta - uv) / sigmas[:, None]) ** 2)), epoch, beta[0]))
    _, epoch, pixel = min(rows, key=lambda row: row[0])
    if interval[0] != interval[1] and epoch in interval:
        return None, "epoch_at_interval_boundary"
    direction = np.linalg.solve(ref[:, :3], np.r_[pixel, 1])
    if abs(direction[2]) < 1e-10:
        return None, "ground_ray_parallel"
    xyz = centers[0] + (POLICY["ball_radius_m"] - centers[0, 2]) / direction[2] * direction
    return xyz[:2], "qualified_model_discrepancy_only"


def estimate(
    frames: ArrayLike,
    pixels: ArrayLike,
    sigmas: ArrayLike,
    cameras: ArrayLike,
    *,
    interval: tuple[float, float],
    fps: float,
) -> dict:
    """Return a qualified fit target or explicit abstention; arrays are not changed.

    Caller supplies only original center training rows within neighboring physical
    event boundaries. Auxiliary windows are attempted only with original support.
    Missing main-wing or
    numerically unresolved support preserves the old fitting target upstream.
    Active vertical bounds and interval boundaries are diagnostics, not evidence
    that the horizontal ground estimate is unusable.
    """
    receipt = dict(policy=deepcopy(POLICY), status="unqualified", interval=list(interval))
    try:
        frames = np.asarray(frames, float)
        pixels = np.asarray(pixels, float)
        sigmas = np.asarray(sigmas, float)
        if np.any(sigmas < 0):
            raise ValueError("negative_native_uncertainty")
        sigmas = np.maximum(sigmas, POLICY["pixel_sigma_floor"])
        cameras = np.asarray(cameras, float)
        lo, hi = map(float, interval)
        if not (np.isfinite([lo, hi, fps]).all() and lo <= hi and fps > 0):
            raise ValueError("invalid_interval_or_cadence")
        receipt["interval_is_exact"] = lo == hi
        if (
            pixels.shape != (len(frames), 2)
            or cameras.shape != (len(frames), 3, 4)
            or sigmas.shape != frames.shape
        ):
            raise ValueError("invalid_native_arrays")
        if not all(np.isfinite(a).all() for a in (frames, pixels, sigmas, cameras)):
            raise ValueError("nonfinite_native_arrays")
        if np.any(np.diff(frames) <= 0):
            raise ValueError("native_frames_not_unique_sorted")
        if np.any(np.linalg.norm(cameras[:, 2, :3], axis=1) == 0):
            raise ValueError("invalid_camera")
        # Normalize arbitrary homogeneous camera scales before conditioning.
        cameras = cameras / np.linalg.norm(cameras[:, 2, :3], axis=1)[:, None, None]
        near = np.abs(frames - (lo + hi) / 2) <= POLICY["maximum_distance_frames"]
        frames, pixels, sigmas, cameras = frames[near], pixels[near], sigmas[near], cameras[near]
        receipt["available_wing_samples"] = [int(np.sum(frames < lo)), int(np.sum(frames > hi))]
        receipt["excluded_interval_rows"] = int(np.sum((frames >= lo) & (frames <= hi)))
        main = _profile(frames, pixels, sigmas, cameras, (lo, hi), fps, 4)
        receipt.update(
            frames=main["frames"],
            best=main["best"],
            profile=main["profile"],
            profile_unconverged_epochs=[
                r["epoch"]
                for r in main["profile"]
                if r["irls_relative_change"] > POLICY["irls_relative_tolerance"]
            ],
            epoch_at_interval_boundary=lo != hi and main["best"]["epoch"] in (lo, hi),
            epoch_steps=len(main["profile"]),
        )
        receipt["main_physical_flag"] = _physical_flag(main, (lo, hi))
        if main["best"]["irls_relative_change"] > POLICY["irls_relative_tolerance"]:
            raise ValueError("irls_unconverged")
        best = main["best"]
        ids = main["ids"]
        dt = (frames[ids] - best["epoch"]) / fps
        incoming = frames[ids] < lo
        z = np.array(best["parameters"])
        p = cameras[ids]
        pred, depth, _ = _project(z, dt, incoming, p)
        projection_jac = (p[:, :2, :3] - pred[:, :, None] * p[:, 2:3, :3]) / depth[:, None, None]
        j = np.zeros((len(ids), 2, 8))
        j[:, :, :2] = projection_jac[:, :, :2]
        j[incoming, :, 2:5] = projection_jac[incoming] * dt[incoming, None, None]
        j[~incoming, :, 5:8] = projection_jac[~incoming] * dt[~incoming, None, None]
        j = (j / sigmas[ids, None, None]).reshape(-1, 8)
        if _condition(j) > POLICY["normalized_condition_limit"]:
            raise ValueError("rank_deficient_or_ill_conditioned")
        dof = 2 * len(ids) - (8 if lo == hi else 9)
        if dof < 1:
            raise ValueError("insufficient_residual_degrees_of_freedom")
        variance_scale = max(1.0, best["chi2"] / dof)
        inv = np.linalg.pinv(j)
        conditional = (inv @ inv.T)[:2, :2] * variance_scale
        xy = z[:2]
        within = [r for r in main["profile"] if r["chi2"] <= best["chi2"] + variance_scale]
        epoch_floor = max(float(np.sum((np.array(r["xyz_m"][:2]) - xy) ** 2)) for r in within)
        auxiliaries = [
            dict(
                wing=wing,
                available=min(receipt["available_wing_samples"]) >= wing,
                status="not_attempted"
                if min(receipt["available_wing_samples"]) >= wing
                else "unavailable",
                **(
                    {"reason": "insufficient_original_rows"}
                    if min(receipt["available_wing_samples"]) < wing
                    else {}
                ),
            )
            for wing in POLICY["auxiliary_wing_samples"]
        ]
        receipt["auxiliaries"] = auxiliaries
        for auxiliary in auxiliaries:
            wing = auxiliary["wing"]
            if not auxiliary["available"]:
                continue
            try:
                other = _profile(frames, pixels, sigmas, cameras, (lo, hi), fps, wing)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                auxiliary.update(status="numerical_failure", reason=str(exc))
                raise
            reason = _physical_flag(other, (lo, hi))
            auxiliary.update(
                status="available", frames=other["frames"], best=other["best"], physical_flag=reason
            )
            if other["best"]["irls_relative_change"] > POLICY["irls_relative_tolerance"]:
                auxiliary.update(status="numerical_failure", reason="irls_unconverged")
                raise ValueError("unqualified_model_window:irls_unconverged")
        available = [r for r in auxiliaries if r["available"]]
        if not available:
            raise ValueError("missing_model_window_check")
        window_floor = max(
            float(np.sum((np.array(r["best"]["xyz_m"][:2]) - xy) ** 2)) for r in available
        )
        quad, quad_reason = _quadratic_ground(frames[ids], pixels[ids], sigmas[ids], p, (lo, hi))
        receipt["curvature_check_missing"] = quad is None
        curvature_floor = 0.0 if quad is None else float(np.sum((quad - xy) ** 2))
        total = conditional + np.eye(2) * (epoch_floor + window_floor + curvature_floor)
        if not np.isfinite(total).all():
            raise ValueError("nonfinite_uncertainty")
        receipt.update(
            status="qualified",
            auxiliaries=auxiliaries,
            uncertainty_kind="conservative_discrepancy_floor_not_calibrated_confidence",
            uncertainty=dict(
                conditional_covariance_m2=conditional.tolist(),
                epoch_profile_floor_m2=epoch_floor,
                model_window_floor_m2=window_floor,
                curvature_model_floor_m2=curvature_floor,
                curvature_status=quad_reason,
                total_covariance_m2=total.tolist(),
                residual_variance_scale=variance_scale,
                residual_degrees_of_freedom=dof,
                floor_availability=dict(
                    epoch=lo != hi, model_window=bool(available), curvature=quad is not None
                ),
                unavailable_model_windows=[r["wing"] for r in auxiliaries if not r["available"]],
                floors_degenerate=bool(
                    epoch_floor + window_floor + curvature_floor
                    <= np.finfo(float).eps * max(1.0, float(np.trace(conditional)))
                ),
            ),
            target=dict(
                xyz_m=best["xyz_m"],
                uncertainty_sigma_m=float(np.sqrt(np.linalg.eigvalsh(total)[-1])),
                source="interval_ballistic_center_fitting_only",
            ),
        )
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        receipt["reason"] = str(exc)
    return receipt


def source_groups(
    scene: model.Scene,
    bounce_events: list[dict],
    boundary_events: list[dict],
    cameras: dict[int, np.ndarray],
    labels: dict[int, np.ndarray],
    radii: dict[int, float] | None,
    *,
    operator: dict,
    camera_distortion: dict[int, np.ndarray] | None = None,
    terminal_rebound_frames=None,
) -> dict:
    """Build fitting receipts from exactly the existing witness partition.

    Neighboring physical intervals fence both wings. No missing camera or ball
    row is interpolated; unsupported source semantics preserve the old target.
    """
    from cv.experiments.connected_shooting.observation_partition import witness_frames
    from cv.pipeline import provenance

    groups = []
    for index, (start, end) in enumerate(zip(scene.contact_frames, scene.contact_frames[1:])):
        group = []
        for event in bounce_events:
            if not start < float(event["frame"]) <= end:
                continue
            receipt = dict(status="unqualified", event=deepcopy(event))
            if (
                operator.get("kind") != "nominal_center"
                or operator.get("exposure_duration_frames") is not None
            ):
                receipt["reason"] = "unsupported_operator"
                group.append(receipt)
                continue
            lo, hi = map(float, event["frame_interval"])
            lower, upper = float(start), float("inf")
            for other in boundary_events:
                if other.get("event_type") not in ("contact", "bounce", "net_hit"):
                    continue
                other_lo, other_hi = map(float, other["frame_interval"])
                if other_hi < lo:
                    lower = max(lower, other_hi)
                if other_lo > hi:
                    upper = min(upper, other_lo)
            frames = sorted(
                int(frame)
                for frame in witness_frames(scene, index, terminal_rebound_frames)
                if float(frame).is_integer()
                and lower < frame < upper
                and frame in cameras
                and frame in labels
                and abs(frame - (lo + hi) / 2) <= POLICY["maximum_distance_frames"]
            )
            defaults = [f for f in frames if f not in (radii or {})]
            receipt.update(
                policy_default_pixel_sigma_input_frames=defaults,
                policy_default_pixel_sigma_input_count=len(defaults),
            )
            if camera_distortion is not None and any(
                frame in camera_distortion and float(camera_distortion[frame][0]) != 0
                for frame in frames
            ):
                receipt["reason"] = "distortion_unsupported"
            elif not frames:
                receipt["reason"] = "insufficient_source_support"
            else:
                receipt.update(
                    estimate(
                        frames,
                        [labels[f] for f in frames],
                        [(radii or {}).get(f, POLICY["pixel_sigma_floor"]) for f in frames],
                        [cameras[f] for f in frames],
                        interval=(lo, hi),
                        fps=scene.fps,
                    )
                )
            receipt["physical_support_bounds"] = [lower, upper if np.isfinite(upper) else None]
            group.append(receipt)
        groups.append(group)
    return dict(
        mode="interval_ballistic_center",
        policy=deepcopy(POLICY),
        implementation=provenance.file_record(Path(__file__)),
        operator=deepcopy(operator),
        groups=groups,
        scope="fitting_seed_and_anchor_only_original_acceptance_witness_unchanged",
        qualified_count=sum(r["status"] == "qualified" for g in groups for r in g),
        preserved_original_count=sum(r["status"] != "qualified" for g in groups for r in g),
    )


def fitting_seed_targets(original: list[np.ndarray | None] | None, receipt: dict) -> list | None:
    """Copy only first-bounce seeds, retaining every original fallback on abstention."""
    if original is None:
        return None
    result = deepcopy(original)
    if len(result) != len(receipt["groups"]):
        raise ValueError("one fit witness group required per flight")
    for index, group in enumerate(receipt["groups"]):
        if group and group[0]["status"] == "qualified":
            result[index] = np.asarray(group[0]["target"]["xyz_m"], float)[:2]
    return result


def fitting_anchor_targets(original: list[list[dict]], receipt: dict) -> list[list[dict]]:
    """Separate deep copy: original acceptance dictionaries are never mutated."""
    result = deepcopy(original)
    if [len(g) for g in result] != [len(g) for g in receipt["groups"]]:
        raise ValueError("one fit witness required per original bounce target")
    for group, witnesses in zip(result, receipt["groups"], strict=True):
        for index, witness in enumerate(witnesses):
            if witness["status"] == "qualified":
                # Do not retain the old graded radius: anchor_plan deliberately
                # converts this heuristic scale using its existing 2*sigma rule.
                group[index] = deepcopy(witness["target"])
    return result
