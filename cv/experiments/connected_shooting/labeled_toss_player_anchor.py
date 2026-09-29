"""Shared soft same-camera player-relative toss conditioning (input-only anchor).

One declared configuration, default off in the pipeline, exact OFF fallback when the
anchor abstains. ``prepare`` builds a player ground anchor from the sided pose row of
each qualified toss frame under that frame's own supported labeled camera; the anchor
is pooled by robust median inside one camera registration window, never across cuts.
``residuals`` adds a dead-zoned Mahalanobis residual on the ballistic toss XY at every
qualified toss row. Both serve paths (joint prefix, isolated n=1) call the same helpers.

Declared, uncalibrated uncertainty: every sigma below is an engineering prior, not a
measured covariance or a confidence interval. No term acts on Vz, apex or direction.
No release event, wrist witness, gate or score is read. Inputs are never mutated.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from cv.experiments.connected_shooting import camera_geometry

SCHEMA = "s6_toss_player_anchor_v1"
GRAVITY = np.array([0.0, 0.0, -9.81])
NATIVE_IMAGE_SIZE = (1920, 1080)
CONSTANTS: dict[str, Any] = dict(
    weight=4.0,
    dead_zone_statures=0.30,
    sigma_floor_m=0.50,
    ankle=dict(
        sigma_px=6.0,
        proxy_height_m=0.08,
        sigma_proxy_height_m=0.06,
        min_joint_confidence=0.5,
    ),
    box=dict(sigma_px=12.0, proxy_height_m=0.0, sigma_proxy_height_m=0.06),
    min_row_confidence=0.25,
    min_frames_per_window=3,
    raw_root_guard_m=2.0,
    spread_warning_m=0.5,
    pixel_step_px=1.0,
    height_step_m=0.01,
)
RAW_ROOT_GUARD_LIMITATION = (
    "compares the labeled-camera anchor with the raw automatic pose calibration root "
    "(court_centre_xy_m), two independently calibrated quantities that disagree by "
    "0.3-0.8 m at the far end; a trip may be a calibration disagreement or a wrong "
    "pose_image_scale rather than a side/association error, and is reported, not repaired"
)
ANKLES = ("left_ankle", "right_ankle")


def ground_point(P, uv, height_m: float, radial: np.ndarray | None = None) -> np.ndarray:
    """Back-project a native pixel onto the horizontal plane Z = height_m (court XY)."""
    P = np.asarray(P, float)
    uv = np.asarray(uv, float)
    if (
        P.shape != (3, 4)
        or uv.shape != (2,)
        or not np.isfinite(P).all()
        or not np.isfinite(uv).all()
    ):
        raise ValueError("finite 3x4 camera and native pixel required")
    uv = camera_geometry.undistort(uv[None], None if radial is None else radial[None])[0]
    q = P[:2, :2] - uv[:, None] * P[2, :2]
    k = uv * (P[2, 2] * height_m + P[2, 3]) - (P[:2, 2] * height_m + P[:2, 3])
    if abs(np.linalg.det(q)) < 1e-12:
        raise ValueError("pixel ray is parallel to the ground plane")
    xy = np.linalg.solve(q, k)
    depth = P[2] @ np.r_[xy, height_m, 1.0]
    if not np.isfinite(xy).all() or depth <= 0:
        raise ValueError("ground point lies behind the camera")
    return xy


def ground_jacobians(
    P, uv, height_m: float, radial: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Central differences: d(xy)/d(pixel) in m/px (2x2) and d(xy)/d(height) in m/m (2,)."""
    step, dz = CONSTANTS["pixel_step_px"], CONSTANTS["height_step_m"]
    uv = np.asarray(uv, float)
    columns = []
    for axis in range(2):
        e = np.zeros(2)
        e[axis] = step
        columns.append(
            (ground_point(P, uv + e, height_m, radial) - ground_point(P, uv - e, height_m, radial))
            / (2 * step)
        )
    J = np.stack(columns, axis=1)
    g = (
        ground_point(P, uv, height_m + dz, radial) - ground_point(P, uv, height_m - dz, radial)
    ) / (2 * dz)
    return J, g


def _finite(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def sided_pose_row(pose_rows, clip: str, frame: int, side: str) -> tuple[dict | None, str]:
    """The unique sided pose row of a native frame at the declared confidence, or why not."""
    name = f"f_{int(frame):04d}.jpg"
    matches = [
        r
        for r in pose_rows
        if r.get("clip") == clip and r.get("frame") == name and r.get("side") == side
    ]
    if len(matches) != 1:
        return None, f"{len(matches)} sided pose rows"
    conf = _finite(matches[0], "conf")
    if conf is None or conf < CONSTANTS["min_row_confidence"]:
        return None, "pose row confidence below declared minimum"
    return matches[0], "supported"


def ankle_pixel(row: dict, scale: float) -> np.ndarray | None:
    """Native ankle midpoint when both ankles exist at the declared joint confidence."""
    values = []
    for joint in ANKLES:
        x, y, c = (_finite(row, f"{joint}_{k}") for k in ("x", "y", "confidence"))
        if x is None or y is None or c is None or c < CONSTANTS["ankle"]["min_joint_confidence"]:
            return None
        values.append([x, y])
    return scale * np.mean(values, axis=0)


def box_bottom_pixel(row: dict, scale: float) -> np.ndarray | None:
    x0, x1, y1 = (_finite(row, k) for k in ("x0", "x1", "y1"))
    if x0 is None or x1 is None or y1 is None:
        return None
    return scale * np.array([(x0 + x1) / 2.0, y1])


def _inside_native(uv: np.ndarray) -> bool:
    w, h = NATIVE_IMAGE_SIZE
    return bool(0.0 <= uv[0] <= w and 0.0 <= uv[1] <= h)


def abstained(reason: str, **extra) -> dict:
    return dict(
        schema=SCHEMA, status="abstained", reason=reason, constants=deepcopy(CONSTANTS), **extra
    )


def _check_space(pose_space, scale: float) -> str | None:
    if pose_space is None:
        return "pose artifact declares no coordinate-space sidecar"
    try:
        artifact = pose_space["artifact_size"]
        image = pose_space["image_size"]
        declared = (float(image["width"]), float(image["height"]))
        scaled = (scale * float(artifact["width"]), scale * float(artifact["height"]))
    except (KeyError, TypeError, ValueError):
        return "pose coordinate-space sidecar lacks artifact/image sizes"
    if declared != tuple(map(float, NATIVE_IMAGE_SIZE)):
        return "pose artifact image space is not the native 1920x1080 frame"
    if any(abs(a - b) > 1e-6 for a, b in zip(scaled, declared)):
        return "pose_image_scale is inconsistent with the declared artifact space"
    return None


def prepare(
    rows,
    cameras: dict | None,
    pose_rows,
    scale: float,
    player: dict | None,
    *,
    clip: str,
    pose_record: dict | None = None,
    pose_space: dict | None = None,
) -> dict:
    """Build the declared player anchor from qualified toss rows; abstain rather than guess.

    ``rows`` are the already qualified incoming rows (same frames the fit will use);
    they are read only. ``cameras`` is the frozen per-frame camera document, used for
    status and registration-window identity; the toss row's own ``camera`` (the same
    P) is the back-projection operator. Radial distortion is unsupported, as for rows.
    """
    frames = [float(r["frame"]) for r in rows]
    if len(frames) != len(set(frames)):
        return abstained("duplicate toss frames")
    if player is None or player.get("side") is None:
        return abstained("no sided serving player")
    stature = _finite(player, "stature_m")
    root = player.get("court_centre_xy_m")
    if stature is None or stature <= 0 or root is None or np.shape(root) != (2,):
        return abstained("serving player lacks finite stature or raw root")
    root = np.asarray(root, float)
    if not np.isfinite(root).all():
        return abstained("serving player lacks finite raw root")
    root_evidence = player.get("root_observation")
    root_receipt = (
        {}
        if root_evidence is None
        else {
            "comparison_root_kind": "same_camera_derived_player_root",
            "original_root_observation": deepcopy(root_evidence),
            "independent_original_root_used_by_guard": False,
        }
    )
    root_limitation = (
        RAW_ROOT_GUARD_LIMITATION
        if root_evidence is None
        else (
            "legacy raw_root fields hold the active same-camera-derived player root; "
            "anchor/box agreement is not independent actor evidence. Original calibration "
            "disagreement is retained separately, without changing this guard numerically"
        )
    )
    if pose_rows is None:
        return abstained("no observed pose rows supplied")
    if not np.isfinite(scale) or scale <= 0:
        return abstained("positive finite pose_image_scale required")
    space_reason = _check_space(pose_space, float(scale))
    if space_reason is not None:
        return abstained(space_reason)
    if cameras is None or "cameras" not in cameras:
        return abstained("no frozen camera document")
    if cameras.get("clip") not in (None, clip):
        return abstained("camera document clip differs from the attempt clip")
    by_frame = {int(c["frame"]): c for c in cameras["cameras"]}
    side = player["side"]
    per_frame, window_of = [], {}
    for row in rows:
        frame = float(row["frame"])
        record = dict(frame=frame)
        cam = by_frame.get(int(frame)) if float(frame).is_integer() else None
        if cam is None or cam.get("status") != "supported" or cam.get("supported") is False:
            record.update(status="unanchored", reason="no supported camera record")
            per_frame.append(record)
            continue
        P = np.asarray(row["camera"], float)
        if P.shape != (3, 4) or not np.array_equal(P, np.asarray(cam.get("P"), float)):
            return abstained(f"toss row {frame:g} camera differs from the frozen camera record")
        if cam.get("local_anchor_frame") is None:
            return abstained(
                f"toss frame {frame:g} lacks a declared camera registration window; "
                "windows cannot be pooled across unknown cuts"
            )
        try:
            radial = camera_geometry.radial_row(row)
            camera_radial = camera_geometry.radial_row(cam)
        except (ValueError, TypeError) as error:
            return abstained(str(error))
        if not (
            (radial is None and camera_radial is None)
            or np.array_equal(radial, camera_radial)
            or (radial is None and camera_radial is not None and camera_radial[0] == 0)
        ):
            return abstained(f"toss frame {frame:g} radial identity differs from frozen camera")
        record["window"] = cam["local_anchor_frame"]
        record["camera_source"] = cam.get("source")
        window_of.setdefault(record["window"], [])
        pose, reason = sided_pose_row(pose_rows, clip, int(frame), side)
        if pose is None:
            record.update(status="unanchored", reason=reason)
            per_frame.append(record)
            continue
        ankle = ankle_pixel(pose, float(scale))
        box = box_bottom_pixel(pose, float(scale))
        if box is None:
            record.update(status="unanchored", reason="pose row lacks a finite box")
            per_frame.append(record)
            continue
        record.update(
            status="anchored",
            pose_confidence=_finite(pose, "conf"),
            pose_court_xy_raw=[_finite(pose, "court_x"), _finite(pose, "court_y")],
            ankle_pixel_native=None if ankle is None else ankle.tolist(),
            box_bottom_pixel_native=box.tolist(),
            camera=P,
            **camera_geometry.radial_fields(cam),
        )
        per_frame.append(record)
        window_of[record["window"]].append(record)
    if any(r["status"] == "unanchored" and "window" not in r for r in per_frame):
        return abstained(
            "a toss row lacks a supported frozen camera record", per_frame=_public(per_frame)
        )
    anchored = [r for r in per_frame if r["status"] == "anchored"]
    if len(anchored) < CONSTANTS["min_frames_per_window"]:
        return abstained(
            f"{len(anchored)} anchored toss frames, fewer than the declared minimum",
            per_frame=_public(per_frame),
        )
    windows = []
    for key, rows_in_window in window_of.items():
        members = [m for m in rows_in_window if m["status"] == "anchored"]
        if len(members) < CONSTANTS["min_frames_per_window"]:
            return abstained(
                f"registration window {key} holds {len(members)} anchored frames, fewer than the "
                "declared minimum; windows are never pooled across cuts",
                per_frame=_public(per_frame),
            )
        kind = (
            "ankle_midpoint"
            if all(m["ankle_pixel_native"] is not None for m in members)
            else "box_bottom_centre"
        )
        spec = CONSTANTS["ankle" if kind == "ankle_midpoint" else "box"]
        z = float(spec["proxy_height_m"])
        points = []
        for m in members:
            uv = np.asarray(
                m["ankle_pixel_native"]
                if kind == "ankle_midpoint"
                else m["box_bottom_pixel_native"],
                float,
            )
            if not _inside_native(uv):
                m.update(status="unanchored", reason="anchor pixel outside the native frame")
                continue
            try:
                xy = ground_point(m["camera"], uv, z, camera_geometry.radial_row(m))
            except ValueError as error:
                m.update(status="unanchored", reason=str(error))
                continue
            m.update(anchor_kind=kind, anchor_pixel_native=uv.tolist(), anchor_xy_m=xy.tolist())
            points.append(xy)
        members = [m for m in members if m["status"] == "anchored"]
        if len(members) < CONSTANTS["min_frames_per_window"]:
            return abstained(
                f"registration window {key} keeps {len(members)} in-frame anchored frames, fewer "
                "than the declared minimum",
                per_frame=_public(per_frame),
            )
        points = np.asarray(points, float)
        centre = np.median(points, axis=0)
        spread = 1.4826 * np.median(np.abs(points - centre), axis=0)
        middle = members[len(members) // 2]
        J, g = ground_jacobians(
            middle["camera"], middle["anchor_pixel_native"], z, camera_geometry.radial_row(middle)
        )
        covariance = (
            J @ J.T * spec["sigma_px"] ** 2
            + np.outer(g, g) * spec["sigma_proxy_height_m"] ** 2
            + np.diag(spread**2)
            + CONSTANTS["sigma_floor_m"] ** 2 * np.eye(2)
        )
        L = np.linalg.cholesky(covariance)
        eigen = np.linalg.eigvalsh(covariance)
        distance = float(np.linalg.norm(centre - root))
        if distance > CONSTANTS["raw_root_guard_m"]:
            return abstained(
                f"anchor lies {distance:.2f} m from the raw pose root, beyond the declared "
                f"{CONSTANTS['raw_root_guard_m']:g} m guard",
                raw_root_guard=dict(
                    limit_m=CONSTANTS["raw_root_guard_m"],
                    distance_m=distance,
                    anchor_xy_m=centre.tolist(),
                    raw_root_xy_m=root.tolist(),
                    limitation=root_limitation,
                    **root_receipt,
                ),
                per_frame=_public(per_frame),
            )
        windows.append(
            dict(
                registration_anchor_frame=key,
                frames=[m["frame"] for m in members],
                anchor_kind=kind,
                sigma_px=spec["sigma_px"],
                proxy_height_m=z,
                sigma_proxy_height_m=spec["sigma_proxy_height_m"],
                anchor_xy_m=centre.tolist(),
                spread_m=spread.tolist(),
                spread_warning=bool(np.any(spread > CONSTANTS["spread_warning_m"])),
                jacobian_frame=middle["frame"],
                jacobian_m_per_px=J.tolist(),
                proxy_height_gradient_m_per_m=g.tolist(),
                covariance_m2=covariance.tolist(),
                sigma_axes_m=np.sqrt(eigen).tolist(),
                whitening=np.linalg.inv(L).tolist(),
                raw_root_distance_m=distance,
            )
        )
    keys = [w["registration_anchor_frame"] for w in windows]
    row_window = [keys.index(r["window"]) for r in per_frame]
    return dict(
        schema=SCHEMA,
        status="supported",
        constants=deepcopy(CONSTANTS),
        clip=clip,
        side=side,
        stature_m=stature,
        dead_zone_m=CONSTANTS["dead_zone_statures"] * stature,
        weight=CONSTANTS["weight"],
        frames=frames,
        row_window=row_window,
        windows=windows,
        registration_windows_declared=True,
        raw_root_xy_m=root.tolist(),
        raw_root_guard=dict(
            limit_m=CONSTANTS["raw_root_guard_m"], limitation=root_limitation, **root_receipt
        ),
        pose_image_scale=float(scale),
        pose_space=deepcopy(pose_space),
        pose_record=deepcopy(pose_record),
        per_frame=_public(per_frame),
        scope=(
            "input-only same-camera player ground anchor; robust median inside one registration "
            "window; sigmas are declared uncalibrated floors, not measured covariance; residual "
            "acts on horizontal toss position only, with a release dead zone; no Vz/apex term"
        ),
    )


def _public(per_frame):
    return [{k: v for k, v in r.items() if k != "camera"} for r in per_frame]


def require_rows(anchor: dict | None, frames) -> None:
    """The anchor must have been prepared from exactly the rows the fit consumes."""
    if anchor is None:
        return
    if anchor.get("status") != "supported":
        raise ValueError("supported player anchor required; pass None for the exact OFF objective")
    if [float(f) for f in anchor["frames"]] != [float(f) for f in frames]:
        raise ValueError("player anchor frames differ from the qualified toss rows")


def ballistic_xy(contact_xyz, incoming, frames, epoch: float, fps: float) -> np.ndarray:
    """Contact-referenced ballistic horizontal positions at the observed frames."""
    dts = (np.asarray(frames, float) - float(epoch)) / float(fps)
    xyz = np.asarray(contact_xyz, float) + np.asarray(incoming, float) * dts[:, None]
    xyz = xyz + 0.5 * GRAVITY * dts[:, None] ** 2
    return xyz[:, :2]


def offsets(anchor: dict, xy) -> np.ndarray:
    """Per-row horizontal offsets from the row's own registration-window anchor (m)."""
    xy = np.asarray(xy, float)
    centres = np.asarray([anchor["windows"][w]["anchor_xy_m"] for w in anchor["row_window"]], float)
    if xy.shape != centres.shape:
        raise ValueError("one horizontal toss position per anchored row required")
    return xy - centres


def residuals(anchor: dict | None, xy) -> np.ndarray:
    """Dead-zoned whitened offsets, weight 4: one declared sigma outside the zone costs 16."""
    if anchor is None:
        return np.empty(0)
    d = offsets(anchor, xy)
    radius = float(anchor["dead_zone_m"])
    norms = np.linalg.norm(d, axis=1)
    factor = np.where(norms > radius, 1.0 - radius / np.maximum(norms, 1e-12), 0.0)
    shrunk = d * factor[:, None]
    out = np.empty_like(shrunk)
    for i, w in enumerate(anchor["row_window"]):
        out[i] = anchor["weight"] * np.asarray(anchor["windows"][w]["whitening"], float) @ shrunk[i]
    return out.ravel()


def report(anchor: dict | None, contact_xyz, incoming, frames, epoch: float, fps: float) -> dict:
    """Receipt of the term at one state: offsets, residuals, costs, player-relative contact."""
    if anchor is None:
        return dict(enabled=False)
    xy = ballistic_xy(contact_xyz, incoming, frames, epoch, fps)
    d = offsets(anchor, xy)
    r = residuals(anchor, xy).reshape(-1, 2)
    contact = np.asarray(contact_xyz, float)
    return dict(
        enabled=True,
        anchor_status=anchor["status"],
        per_row=[
            dict(
                frame=float(f),
                window=int(w),
                toss_xy_m=xy[i].tolist(),
                offset_xy_m=d[i].tolist(),
                offset_norm_m=float(np.linalg.norm(d[i])),
                inside_dead_zone=bool(np.linalg.norm(d[i]) <= anchor["dead_zone_m"]),
                residual=r[i].tolist(),
                cost=float(r[i] @ r[i]),
            )
            for i, (f, w) in enumerate(zip(frames, anchor["row_window"], strict=True))
        ],
        anchor_cost=float(np.sum(r**2)),
        player_relative_contact_m=[
            (contact[:2] - np.asarray(w["anchor_xy_m"], float)).tolist() for w in anchor["windows"]
        ],
        contact_height_statures=float(contact[2] / anchor["stature_m"]),
    )


def incoming_diagnostics(contact_xyz, incoming, frames, epoch: float, fps: float) -> dict:
    """Vz at the observed frames, apex epoch and contact sign (root's naming), no term."""
    incoming = np.asarray(incoming, float)
    dts = (np.asarray(frames, float) - float(epoch)) / float(fps)
    vz = incoming[2] + GRAVITY[2] * dts
    return dict(
        incoming_vz_at_observed_frames=vz.tolist(),
        observed_rows_all_rising=bool(np.all(vz > 0)),
        apex_native_frame=float(epoch + fps * incoming[2] / (-GRAVITY[2])),
        incoming_vz_sign_at_contact=("rising" if incoming[2] > 0 else "descending"),
        incoming_horizontal_speed_mps=float(np.linalg.norm(incoming[:2])),
    )
