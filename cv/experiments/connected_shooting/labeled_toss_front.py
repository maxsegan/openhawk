"""Original paired-axis incoming toss front operator.

Research observation model, not a shutter measurement. Enriches a copy of
existing toss rows with the original leading-minus-trailing image axis. Predicts
the visible leading extent by reusing swept_exposure.directional_tips over each
native [frame, frame+duration] ballistic sample, the original per-frame camera,
existing ball radius and the outgoing operator's 1.5px blur allowance.
Unresolved apex rows keep the native-frame center proxy and original
uncertainty; no direction is inferred. This module does not optimize or score.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from cv.experiments.connected_shooting import camera_geometry, swept_exposure, toss_witness
from cv.experiments.connected_shooting.model import R_BALL

GRAVITY = np.array([0.0, 0.0, -9.81])
BLUR_RADIUS_PX = 1.5
BALLISTIC_SAMPLES = 9
CENTER = "center"
PAIRED_SWEPT_FRONT = "paired_swept_front"
UNRESOLVED_AXIS_CENTER_PROXY = "unresolved-axis center proxy"
PAIRED_OR_MOTION_SWEPT_FRONT = "paired_or_motion_swept_front"
CANDIDATE_MOTION_SWEPT_FRONT = "candidate-motion swept front"
MODELS = (CENTER, PAIRED_SWEPT_FRONT, PAIRED_OR_MOTION_SWEPT_FRONT)

BLUR_SEMANTICS = (
    "paired directional_tips over [native frame, frame+duration] with original "
    "per-frame P, existing ball radius and outgoing 1.5px isotropic blur "
    "allowance; duration/blur are inherited observation-model conditions, not "
    "measured shutter/radiance parameters; unresolved-axis rows keep the "
    "native-frame center proxy and original uncertainty"
)
CENTER_SEMANTICS = (
    "ballistic sample projected at the native frame; not a shutter or swept-front model"
)


def assert_incoming_model(model: str, duration: float | None) -> None:
    if model not in MODELS:
        raise ValueError(f"incoming_toss_model must be one of {MODELS}")
    if model == CENTER:
        return
    if (
        duration is None
        or isinstance(duration, (bool, np.bool_))
        or not np.isfinite(float(duration))
        or float(duration) <= 0
    ):
        raise ValueError("paired_swept_front requires a positive finite native exposure duration")


def require_enriched(rows: list[dict]) -> None:
    for row in rows:
        operator = row.get("incoming_operator")
        if operator == CENTER:
            if (
                row.get("source") != toss_witness.AUTOMATIC_CENTER_SOURCE
                or not toss_witness.qualified_automatic_center(row)
                or row.get("axis") is not None
            ):
                raise ValueError("native center operator requires qualified automatic observations")
            continue
        if operator == UNRESOLVED_AXIS_CENTER_PROXY:
            if row.get("axis") is not None:
                raise ValueError("unresolved-axis center proxy must not carry a derived image axis")
            continue
        if operator != PAIRED_SWEPT_FRONT:
            raise ValueError(
                "paired_swept_front requires original paired or unresolved-axis operator identities"
            )
        axis = np.asarray(row.get("axis"), float)
        if axis.shape != (2,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
            raise ValueError("finite original paired leading-minus-trailing axis required")


def enrich(observations: dict, labels: dict, clip: str) -> dict:
    """Copy observation rows and attach original paired axes; never mutate inputs."""
    if observations.get("clip") not in (None, clip):
        raise ValueError("observation clip disagrees with supplied original clip")
    groups = [record for record in labels["ball"]["records"] if record["clip"] == clip]
    if len(groups) != 1:
        raise ValueError("one original clip ball record required")
    points = {}
    for point in groups[0]["frames"]:
        frame = float(point["frame"])
        if frame in points:
            raise ValueError("unique original native records required")
        points[frame] = point
    result = deepcopy(observations)
    for row in result["rows"]:
        point = points.get(float(row["frame"]))
        if point is None or point.get("status") != "visible":
            raise ValueError("observation frame lacks a visible original ball label")
        leading = [point["x1080"], point["y1080"]]
        if not np.array_equal(row["pixel"], leading):
            raise ValueError("original leading coordinates do not match observation pixel")
        if point.get("annotation_origin") == "automatic":
            if (
                not toss_witness.qualified_automatic_center(point)
                or row.get("source") != toss_witness.AUTOMATIC_CENTER_SOURCE
                or not toss_witness.qualified_automatic_center(row)
            ):
                raise ValueError("qualified original native detector center required")
            row["incoming_operator"] = CENTER
            row["axis"] = None
            continue
        streak = point.get("streak") or {}
        streak_leading = streak.get("leading")
        if streak_leading is not None and not np.array_equal(
            [streak_leading["x1080"], streak_leading["y1080"]], leading
        ):
            raise ValueError("streak leading differs from original labeled front")
        trailing = streak.get("trailing")
        if trailing is None:
            row["incoming_operator"] = UNRESOLVED_AXIS_CENTER_PROXY
            row["axis"] = None
            continue
        if streak.get("status") != "paired":
            raise ValueError("original trailing tip requires declared paired support")
        axis = np.asarray(leading, float) - np.asarray(
            [trailing["x1080"], trailing["y1080"]], float
        )
        if not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
            raise ValueError("finite distinct original paired tips required")
        row["incoming_operator"] = PAIRED_SWEPT_FRONT
        row["axis"] = axis.tolist()
    return result


def ballistic_curve(contact_xyz, velocity, contact_epoch, fps, frame, duration):
    times = np.linspace(float(frame), float(frame) + float(duration), BALLISTIC_SAMPLES)
    dt = (times - contact_epoch) / fps
    contact_xyz = np.asarray(contact_xyz, float)
    velocity = np.asarray(velocity, float)
    return contact_xyz + dt[:, None] * velocity + 0.5 * dt[:, None] ** 2 * GRAVITY


def paired_leading_front(row, contact_xyz, velocity, contact_epoch, fps, duration):
    xyz = ballistic_curve(contact_xyz, velocity, contact_epoch, fps, row["frame"], duration)
    tips, _ = swept_exposure.directional_tips(
        xyz,
        np.asarray(row["camera"], float),
        np.asarray(row["axis"], float),
        blur_radius_px=BLUR_RADIUS_PX,
        ball_radius_m=R_BALL,
        radial=camera_geometry.radial_row(row),
    )
    return tips[1]


def overlay_paired_leading(
    rows,
    center_predictions,
    contact_xyz,
    velocity,
    contact_epoch,
    fps,
    duration,
):
    """Replace paired-row center projections with swept leading fronts.

    Unresolved-axis rows keep the supplied native-frame center predictions.
    """
    assert_incoming_model(PAIRED_SWEPT_FRONT, duration)
    require_enriched(rows)
    predicted = np.asarray(center_predictions, float).copy()
    if predicted.shape != (len(rows), 2) or not np.isfinite(predicted).all():
        raise ValueError("one finite center prediction per incoming toss row required")
    for i, row in enumerate(rows):
        if row["incoming_operator"] in (CENTER, UNRESOLVED_AXIS_CENTER_PROXY):
            continue
        predicted[i] = paired_leading_front(
            row, contact_xyz, velocity, contact_epoch, fps, duration
        )
    return predicted


def operator_identity(row: dict, model: str) -> str:
    if model == CENTER:
        return CENTER
    return row["incoming_operator"]


def policy(model: str, duration: float | None) -> dict:
    assert_incoming_model(model, duration)
    paired = model != CENTER
    return dict(
        incoming_toss_model=model,
        incoming_exposure_duration_frames=None if duration is None else float(duration),
        incoming_blur_radius_px=BLUR_RADIUS_PX if paired else None,
        incoming_ballistic_samples=BALLISTIC_SAMPLES if paired else None,
        incoming_ball_radius_m=R_BALL if paired else None,
        incoming_blur_semantics=(
            BLUR_SEMANTICS + "; missing axes use candidate projected exposure chord only "
            "where a frozen three-row input window has agreeing consecutive motion; "
            "undefined candidate chord refuses the trial, never drops observations"
            if model == PAIRED_OR_MOTION_SWEPT_FRONT
            else BLUR_SEMANTICS
            if paired
            else CENTER_SEMANTICS
        ),
    )


def qualify_motion_rows(rows: list[dict]) -> list[dict]:
    """Freeze monotonic qualification from already eligible native input rows.

    Three-row windows are centered except at support endpoints. Original paired
    axes and all original uncertainties/pixels/cameras remain unchanged. No
    candidate state, residual, or labels outside eligible support enter this rule.
    """
    require_enriched(rows)
    result = deepcopy(rows)
    frames = np.asarray([r["frame"] for r in rows], float)
    pixels = np.asarray([r["pixel"] for r in rows], float)
    if not np.isfinite(frames).all() or np.any(np.diff(frames) <= 0):
        raise ValueError("motion qualification requires increasing unique native frames")
    if pixels.shape != (len(rows), 2) or not np.isfinite(pixels).all():
        raise ValueError("motion qualification requires finite original input pixels")
    for i, row in enumerate(result):
        if row["incoming_operator"] != UNRESOLVED_AXIS_CENTER_PROXY:
            continue
        receipt = dict(status="insufficient_input_support", window_frames=[])
        if len(rows) >= 3:
            start = min(max(i - 1, 0), len(rows) - 3)
            times = frames[start : start + 3]
            speeds = np.diff(pixels[start : start + 3], axis=0) / np.diff(times)[:, None]
            norms = np.linalg.norm(speeds, axis=1)
            dot = float(speeds[0] @ speeds[1])
            supported = bool(np.all(norms > 1e-9) and dot > 0)
            receipt = dict(
                status="monotonic_input_motion" if supported else "unresolved_input_turn",
                window_frames=times.tolist(),
                input_velocity_px_per_frame=speeds.tolist(),
                velocity_dot_product=dot,
            )
            if supported:
                row["incoming_operator"] = CANDIDATE_MOTION_SWEPT_FRONT
        row["motion_qualification"] = receipt
    return result


def motion_leading_front(row, contact_xyz, velocity, contact_epoch, fps, duration):
    """Predict directional front from candidate chord, never from image residual."""
    xyz = ballistic_curve(contact_xyz, velocity, contact_epoch, fps, row["frame"], duration)
    camera = np.asarray(row["camera"], float)
    radial = camera_geometry.radial_row(row)
    projected = camera_geometry.project(
        np.repeat(camera[None], 2, axis=0),
        xyz[[0, -1]],
        None if radial is None else np.repeat(radial[None], 2, axis=0),
    )
    axis = projected[-1] - projected[0]
    if not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
        raise ValueError("ambiguous numerically zero candidate exposure chord")
    tips, _ = swept_exposure.directional_tips(
        xyz, camera, axis, blur_radius_px=BLUR_RADIUS_PX, ball_radius_m=R_BALL, radial=radial
    )
    return tips[1]


def overlay_motion_leading(
    rows, center_predictions, contact_xyz, velocity, contact_epoch, fps, duration
):
    """Optional mixed measured/candidate-axis model with fixed input eligibility."""
    assert_incoming_model(PAIRED_OR_MOTION_SWEPT_FRONT, duration)
    predicted = np.asarray(center_predictions, float).copy()
    if predicted.shape != (len(rows), 2) or not np.isfinite(predicted).all():
        raise ValueError("one finite center prediction per incoming toss row required")
    for i, row in enumerate(rows):
        if row.get("incoming_operator") == CANDIDATE_MOTION_SWEPT_FRONT:
            if (
                row.get("axis") is not None
                or row.get("motion_qualification", {}).get("status") != "monotonic_input_motion"
            ):
                raise ValueError("candidate motion requires frozen original monotonic support")
            predicted[i] = motion_leading_front(
                row, contact_xyz, velocity, contact_epoch, fps, duration
            )
        else:
            require_enriched([row])
            if row["incoming_operator"] == PAIRED_SWEPT_FRONT:
                predicted[i] = paired_leading_front(
                    row, contact_xyz, velocity, contact_epoch, fps, duration
                )
    return predicted
