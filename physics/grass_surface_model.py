"""Model-free broadcast measurements and the regional-v2 grass calibration.

This module is evaluation-side calibration code.  It reads the human-labeled attempts only
when run as a command; automatic inference never imports or opens those labels.  The emitted
``measurements.json`` records native image motion on both sides of every labeled bounce,
the native bounce-to-next-contact time, the ray-fixed landing region and line distance.

``regional_v2_model`` is the small runtime result: hard-court measured coefficients are the
prior centre, so missing broadcast evidence is exactly the current grass reference.  The only
departures are three declared per-broadcast scalars: pace/wear ``w``, fixed-kernel amplitude
``a`` and a line-skid switch.  Human-derived fitted rows are confined to the labeled-attempt
experiment and are never a production default.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np

from physics import surface_model

BALL_CENTRE_HEIGHT_M = 0.033
SIDE_PICTURES = 4
MIN_SIDE_PICTURES = 2
MAX_SIDE_SPAN_FRAMES = 8
GRASS_ATTEMPTS = (
    "grassatp2024r16_pt0001_a01",
    "grassatp2024r16_pt0003_a01",
    "grassatp2024r16_pt0004_a01",
    "wim2019f_pt0002_a01",
    "wim2019f_pt0004_a01",
    "wim2023f_pt0003_a01",
    "wim2023f_pt0005_a01",
    "wim2024f_pt0003_a01",
    "wim2024f_pt0004_a01",
    "wim2025f_pt0002_a01",
    "wim2025f_pt0003_a01",
    "wim2025f_pt0004_a01",
    "wim2025r32_pt0001_a01",
    "wim2025r32_pt0002_a01",
)
EXPECTED_ATTEMPTS = {"grass": 14, "hard": 51, "clay": 24}

# Frozen from ``processed/grassmodel/measurement/measurements.json``.  These are opened,
# human-derived development calibrations, not automatic surface inference.  An absent broadcast
# is deliberately the exact hard-profile prior.
REGIONAL_V2_BROADCASTS: dict[str, dict[str, Any]] = {
    "atp_2024_540_r16_217_taylor_fritz_alexander_zverev": {
        "wear": 1.001275,
        "amplitude": -0.027103,
        "line_switch": False,
        "evidence_bounces": 8,
    },
    "wim2019f_w_halep_williams": {
        "wear": 1.006989,
        "amplitude": -0.027630,
        "line_switch": False,
        "evidence_bounces": 4,
    },
    "wim2023f_m_alcaraz_djokovic": {
        "wear": 0.988403,
        "amplitude": 0.017523,
        "line_switch": False,
        "evidence_bounces": 3,
    },
    "wim2024f_w_krejcikova_paolini": {
        "wear": 0.993274,
        "amplitude": 0.016468,
        "line_switch": False,
        "evidence_bounces": 5,
    },
    "wim2025f_m_sinner_alcaraz": {
        "wear": 0.980960,
        "amplitude": 0.009245,
        "line_switch": False,
        "evidence_bounces": 8,
    },
    "wim2025r32_w_sabalenka_raducanu": {
        "wear": 0.998044,
        "amplitude": -0.024875,
        "line_switch": False,
        "evidence_bounces": 3,
    },
}


def regional_v2_model(match_id: str | None) -> surface_model.SurfaceModel:
    """Return this grass broadcast's three-scalar model around the exact hard prior."""
    row = REGIONAL_V2_BROADCASTS.get(str(match_id), {})
    return surface_model.SurfaceModel(
        surface="grass",
        wear=float(row.get("wear", 1.0)),
        amplitude=float(row.get("amplitude", 0.0)),
        line_switch=bool(row.get("line_switch", False)),
        base_source="hard_reference",
        provenance=(
            "opened_labeled_broadcast_model_free_regional_v2"
            if row.get("evidence_bounces", 0)
            else "hard_reference_prior_no_broadcast_evidence"
        ),
    )


def tournament_week(match_id: str) -> str:
    """Week bucket declared by the match id, without consulting a fit result.

    Grand-slam R16/QF/SF/F are week two; R128/R64/R32 are week one.  Tour events
    (including the ATP 500 ``grassatp`` broadcast) are one-week events.
    """
    key = str(match_id).lower()
    if not key.startswith(("ao", "rg", "uso", "wim")):
        return "week_1"
    token = key.split("_")[0]
    return "week_2" if re.search(r"(?:r16|qf|sf|f)$", token) else "week_1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_record(root: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"revision": revision, "dirty": dirty}


def _camera_ray(projection: np.ndarray, pixel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(projection[:, :3], dtype=float)
    offset = np.asarray(projection[:, 3], dtype=float)
    centre = -np.linalg.solve(matrix, offset)
    direction = np.linalg.solve(matrix, np.asarray([pixel[0], pixel[1], 1.0]))
    direction /= float(np.linalg.norm(direction))
    return centre, direction


def _ground_point(projection: np.ndarray, pixel: np.ndarray) -> np.ndarray:
    centre, direction = _camera_ray(projection, pixel)
    if abs(float(direction[2])) < 1e-9:
        raise ValueError("ray parallel to court")
    distance = (BALL_CENTRE_HEIGHT_M - float(centre[2])) / float(direction[2])
    if distance <= 0.0:
        raise ValueError("court behind camera")
    return centre + direction * distance


def _event_kind(row: dict[str, Any]) -> str:
    return str(row.get("event_type") or row.get("kind") or row.get("type") or "")


def _native_seconds(frame: float, labels: dict[int, dict[str, Any]]) -> float | None:
    """Interpolate an event epoch on actual labeled native PTS; never synthesize cadence."""
    values = sorted(
        (number, float(row["native_pts_seconds"]))
        for number, row in labels.items()
        if row.get("native_pts_seconds") is not None
        and math.isfinite(float(row["native_pts_seconds"]))
    )
    if not values:
        return None
    frames = np.asarray([row[0] for row in values], dtype=float)
    seconds = np.asarray([row[1] for row in values], dtype=float)
    index = int(np.searchsorted(frames, frame))
    if index == 0 or index == len(frames):
        return None
    left, right = index - 1, index
    if frames[right] == frames[left]:
        return None
    fraction = (frame - frames[left]) / (frames[right] - frames[left])
    return float(seconds[left] + fraction * (seconds[right] - seconds[left]))


def _side_frames(
    bounce_frame: float,
    previous_frame: float,
    next_frame: float,
    labels: dict[int, dict[str, Any]],
    cameras: dict[int, np.ndarray],
) -> tuple[list[int], list[int]] | None:
    floor, ceil = math.floor(bounce_frame), math.ceil(bounce_frame)
    pre_last = floor - 1 if floor == ceil else floor
    post_first = floor + 1 if floor == ceil else ceil
    pre = [
        frame
        for frame in range(pre_last - MAX_SIDE_SPAN_FRAMES + 1, pre_last + 1)
        if frame > previous_frame
        and frame in labels
        and frame in cameras
        and labels[frame].get("native_pts_seconds") is not None
    ]
    post = [
        frame
        for frame in range(post_first, post_first + MAX_SIDE_SPAN_FRAMES)
        if frame < next_frame
        and frame in labels
        and frame in cameras
        and labels[frame].get("native_pts_seconds") is not None
    ]
    if len(pre) < MIN_SIDE_PICTURES or len(post) < MIN_SIDE_PICTURES:
        return None
    return pre[-SIDE_PICTURES:], post[:SIDE_PICTURES]


def _pixel_motion(frames: Sequence[int], labels: dict[int, dict[str, Any]]) -> dict[str, Any]:
    seconds = np.asarray([float(labels[frame]["native_pts_seconds"]) for frame in frames])
    pixels = np.asarray(
        [[float(labels[frame]["x1080"]), float(labels[frame]["y1080"])] for frame in frames]
    )
    centred = seconds - float(np.mean(seconds))
    denominator = float(centred @ centred)
    if denominator <= 0.0:
        raise ValueError("non-distinct native timestamps")
    velocity = centred @ pixels / denominator
    intercept = np.mean(pixels, axis=0)
    residual = pixels - (intercept + centred[:, None] * velocity)
    return {
        "frames": list(frames),
        "native_pts_seconds": seconds.tolist(),
        "pixels": pixels.tolist(),
        "velocity_px_per_s": velocity.tolist(),
        "speed_px_per_s": float(np.linalg.norm(velocity)),
        "linear_rms_px": float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
        "mean_frame": float(np.mean(frames)),
        "mean_seconds": float(np.mean(seconds)),
        "intercept_pixel": intercept.tolist(),
    }


def _pixel_at_seconds(motion: dict[str, Any], seconds: float) -> np.ndarray:
    return np.asarray(motion["intercept_pixel"], dtype=float) + (
        seconds - float(motion["mean_seconds"])
    ) * np.asarray(motion["velocity_px_per_s"], dtype=float)


def _nearest_camera(cameras: dict[int, np.ndarray], frame: float) -> np.ndarray:
    return cameras[min(cameras, key=lambda value: abs(float(value) - frame))]


def measure_bounce(
    *,
    key: str,
    match_id: str,
    surface: str,
    events: Sequence[dict[str, Any]],
    index: int,
    labels: dict[int, dict[str, Any]],
    cameras: dict[int, np.ndarray],
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any] | None]:
    event = events[index]
    frame = float(event["frame"])
    previous = float(events[index - 1]["frame"]) if index else -math.inf
    following = float(events[index + 1]["frame"]) if index + 1 < len(events) else math.inf
    selected = _side_frames(frame, previous, following, labels, cameras)
    if selected is None:
        return None, "fewer_than_two_native_pts_fronts_on_one_side", None
    bounce_seconds = _native_seconds(frame, labels)
    if bounce_seconds is None:
        return None, "bounce_not_bracketed_by_native_pts", None
    try:
        incoming = _pixel_motion(selected[0], labels)
        outgoing = _pixel_motion(selected[1], labels)
        impact_pixels = [
            _pixel_at_seconds(incoming, bounce_seconds),
            _pixel_at_seconds(outgoing, bounce_seconds),
        ]
        camera = _nearest_camera(cameras, frame)
        impact_points = [_ground_point(camera, pixel) for pixel in impact_pixels]
    except (ValueError, np.linalg.LinAlgError):
        return None, "motion_or_ground_ray_failed", None
    anchor = np.mean(impact_points, axis=0)
    incoming_velocity = np.asarray(incoming["velocity_px_per_s"], dtype=float)
    outgoing_velocity = np.asarray(outgoing["velocity_px_per_s"], dtype=float)
    denominator = float(np.linalg.norm(incoming_velocity) * np.linalg.norm(outgoing_velocity))
    cosine = float(np.dot(incoming_velocity, outgoing_velocity) / max(denominator, 1e-12))
    direction_change = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
    next_contact = next(
        (candidate for candidate in events[index + 1 :] if _event_kind(candidate) == "contact"),
        None,
    )
    next_contact_seconds = (
        _native_seconds(float(next_contact["frame"]), labels) if next_contact is not None else None
    )
    line_distance, line_name = surface_model.line_distance_m(float(anchor[0]), float(anchor[1]))
    row = {
        "attempt_key": key,
        "match_id": match_id,
        "surface": surface,
        "tournament_week": tournament_week(match_id),
        "bounce_frame": frame,
        "bounce_native_pts_seconds": bounce_seconds,
        "landing_x_m": float(anchor[0]),
        "landing_y_m": float(anchor[1]),
        "landing_region": surface_model.landing_region(float(anchor[0]), float(anchor[1])),
        "region_kernel": surface_model.region_kernel(float(anchor[0]), float(anchor[1])),
        "line_distance_m": float(line_distance),
        "nearest_line": line_name,
        "on_line": bool(line_distance <= surface_model.LINE_SWITCH_M),
        "impact_ray_disagreement_m": float(np.linalg.norm(impact_points[0] - impact_points[1])),
        "pre_speed_px_per_s": incoming["speed_px_per_s"],
        "post_speed_px_per_s": outgoing["speed_px_per_s"],
        "post_pre_speed_ratio": outgoing["speed_px_per_s"] / incoming["speed_px_per_s"],
        "direction_change_deg": direction_change,
        "bounce_to_next_contact_seconds": (
            None if next_contact_seconds is None else next_contact_seconds - bounce_seconds
        ),
        "next_contact_frame": None if next_contact is None else float(next_contact["frame"]),
        "incoming_linear_rms_px": incoming["linear_rms_px"],
        "outgoing_linear_rms_px": outgoing["linear_rms_px"],
        "incoming_frames": incoming["frames"],
        "outgoing_frames": outgoing["frames"],
    }
    plot = {"row": row, "incoming": incoming, "outgoing": outgoing, "impact_pixels": impact_pixels}
    return row, None, plot


def _summary(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in fields:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row[field])].append(row)
        output[field] = {}
        for key, values in sorted(grouped.items()):
            output[field][key] = {
                "n": len(values),
                "post_pre_speed_ratio_mean": float(
                    np.mean([row["post_pre_speed_ratio"] for row in values])
                ),
                "direction_change_deg_mean": float(
                    np.mean([row["direction_change_deg"] for row in values])
                ),
                "bounce_to_next_contact_seconds_mean": (
                    float(
                        np.mean(
                            [
                                row["bounce_to_next_contact_seconds"]
                                for row in values
                                if row["bounce_to_next_contact_seconds"] is not None
                            ]
                        )
                    )
                    if any(row["bounce_to_next_contact_seconds"] is not None for row in values)
                    else None
                ),
                "bounce_to_next_contact_n": sum(
                    row["bounce_to_next_contact_seconds"] is not None for row in values
                ),
            }
    return output


def _usable_measurement(row: dict[str, Any]) -> bool:
    """Model-free quality gate fixed before fitting any grass broadcast scalar."""
    return (
        row["impact_ray_disagreement_m"] <= 0.75
        and row["incoming_linear_rms_px"] <= 6.0
        and row["outgoing_linear_rms_px"] <= 6.0
        and 0.2 <= row["post_pre_speed_ratio"] <= 5.0
    )


def fit_regional_v2(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Fit conservative image-space departures from the exact hard-profile prior.

    The response is log post/pre pixel speed, centred on hard controls in the same landing
    region.  A two-parameter ridge fit supplies a broadcast intercept and fixed-kernel slope.
    Their physical mapping is deliberately shrunk by the observed hard-control scale: one
    standard deviation can move ``w`` or ``a`` by at most 0.04.  With no usable contrast both
    parameters are exactly their hard-reference prior.  The line switch requires at least two
    stable near-line bounces and a positive skid contrast; otherwise it remains off.
    """

    if not rows:
        return {}

    hard = [row for row in rows if row["surface"] == "hard" and _usable_measurement(row)]
    if not hard:
        raise ValueError("regional-v2 needs quality-gated hard controls")
    by_region: dict[str, list[float]] = defaultdict(list)
    for row in hard:
        by_region[str(row["landing_region"])].append(math.log(row["post_pre_speed_ratio"]))
    all_hard = [math.log(row["post_pre_speed_ratio"]) for row in hard]
    hard_mean = float(np.mean(all_hard))
    hard_sd = max(float(np.std(all_hard)), 0.10)
    controls = {region: float(np.mean(values)) for region, values in by_region.items() if values}
    output: dict[str, dict[str, Any]] = {}
    broadcasts = sorted({str(row["match_id"]) for row in rows if row["surface"] == "grass"})
    for match_id in broadcasts:
        all_values = [
            row for row in rows if row["surface"] == "grass" and row["match_id"] == match_id
        ]
        values = [row for row in all_values if _usable_measurement(row)]
        design, target = [], []
        for row in values:
            design.append([1.0, float(row["region_kernel"])])
            target.append(
                math.log(row["post_pre_speed_ratio"])
                - controls.get(str(row["landing_region"]), hard_mean)
            )
        if not target:
            output[match_id] = {
                "wear": 1.0,
                "amplitude": 0.0,
                "line_switch": False,
                "evidence_bounces": 0,
                "reason": "no_model_free_measurement",
            }
            continue
        matrix = np.asarray(design, dtype=float)
        vector = np.asarray(target, dtype=float)
        # Prior SD is one hard-control SD for the intercept and twice that for the regional
        # slope.  It keeps single-region broadcasts close to zero instead of aliasing intercept
        # into amplitude.
        penalty = np.diag([1.0 / hard_sd**2, 1.0 / (2.0 * hard_sd) ** 2])
        solved = np.linalg.solve(
            matrix.T @ matrix / hard_sd**2 + penalty, matrix.T @ vector / hard_sd**2
        )
        scale = 0.04 / hard_sd
        wear = float(np.clip(1.0 + solved[0] * scale, 0.96, 1.04))
        amplitude = float(np.clip(solved[1] * scale, -0.04, 0.04))
        near_line = [row for row in values if row["line_distance_m"] <= 0.15]
        off_line = [row for row in values if row["line_distance_m"] > 0.15]
        line_contrast = None
        if len(near_line) >= 2 and off_line:
            line_contrast = float(
                np.mean([math.log(row["post_pre_speed_ratio"]) for row in near_line])
                - np.mean([math.log(row["post_pre_speed_ratio"]) for row in off_line])
            )
        output[match_id] = {
            "wear": wear,
            "amplitude": amplitude,
            "line_switch": bool(line_contrast is not None and line_contrast > 0.10),
            "line_contrast_log_speed_ratio": line_contrast,
            "evidence_bounces": len(values),
            "measured_bounces_before_quality_gate": len(all_values),
            "hard_control_log_ratio_sd": hard_sd,
            "model_free_intercept": float(solved[0]),
            "model_free_kernel_slope": float(solved[1]),
        }
    return output


def _validate_frozen_profiles(profiles: dict[str, dict[str, Any]]) -> None:
    """Fail closed when the checked-in runtime scalars do not match this calibration."""
    if set(profiles) != set(REGIONAL_V2_BROADCASTS):
        raise ValueError("the fitted and checked-in regional-v2 broadcast sets differ")
    for match_id, fitted in profiles.items():
        frozen = REGIONAL_V2_BROADCASTS[match_id]
        for field in ("wear", "amplitude"):
            if not math.isclose(float(fitted[field]), float(frozen[field]), abs_tol=1e-6):
                raise ValueError(f"stale regional-v2 {field} for {match_id}")
        for field in ("line_switch", "evidence_bounces"):
            if fitted[field] != frozen[field]:
                raise ValueError(f"stale regional-v2 {field} for {match_id}")


def _video_path(data_root: Path, match_id: str, clip: str) -> tuple[Path, bool] | None:
    point = clip.split("__")[-1]
    # This one shipped pt0003 clip ends before four later labeled bounces.  The immutable
    # labels carry absolute native PTS, so use the source broadcast for every picture in that
    # attempt instead of pretending the short clip contains those exposures.
    if match_id == "wim2025f_m_sinner_alcaraz" and point == "pt0003":
        sources = sorted(
            (data_root / "videos").glob("wim2025f_m_sinner_alcaraz*.mp4")
        )
        if sources:
            return sources[0], True
    roots = sorted((data_root / "videos").glob("clips_native"))
    roots += sorted((data_root / "videos").glob("clips_native_*"))
    for root in roots:
        candidate = root / match_id / f"{point}.mp4"
        if candidate.is_file():
            return candidate, False
    return None


def _render_native(
    path: Path,
    video: tuple[Path, bool],
    diagnostic: dict[str, Any],
    labels: dict[int, dict[str, Any]],
    plot: dict[str, Any] | None,
    fallback_image: Path | None,
) -> None:
    import cv2

    row = plot["row"] if plot is not None else diagnostic
    video_path, absolute_pts = video
    if absolute_pts:
        native_seconds = row.get("bounce_native_pts_seconds")
        if native_seconds is None:
            native_seconds = _native_seconds(float(row["bounce_frame"]), labels)
        if native_seconds is None:
            image = cv2.imread(str(fallback_image)) if fallback_image is not None else None
        else:
            with tempfile.NamedTemporaryFile(suffix=".jpg") as temporary:
                extracted = subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-ss",
                        f"{float(native_seconds):.6f}",
                        "-i",
                        str(video_path),
                        "-frames:v",
                        "1",
                        "-y",
                        temporary.name,
                    ],
                    check=False,
                )
                image = cv2.imread(temporary.name) if extracted.returncode == 0 else None
    else:
        capture = cv2.VideoCapture(str(video_path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(row["bounce_frame"])) - 1))
        ok, image = capture.read()
        capture.release()
        if not ok:
            image = None
    if image is None:
        return
    if plot is None:
        frame = float(row["bounce_frame"])
        pre = sorted(value for value in labels if frame - 8 <= value < frame)[-SIDE_PICTURES:]
        post = sorted(value for value in labels if frame < value <= frame + 8)[:SIDE_PICTURES]
        motions = []
        for frames in (pre, post):
            motions.append(
                {
                    "pixels": [
                        [float(labels[value]["x1080"]), float(labels[value]["y1080"])]
                        for value in frames
                    ]
                }
            )
    else:
        motions = [plot["incoming"], plot["outgoing"]]
    for motion, color in zip(motions, ((255, 100, 0), (0, 150, 255)), strict=True):
        pixels = [tuple(int(round(value)) for value in point) for point in motion["pixels"]]
        for first, second in zip(pixels, pixels[1:]):
            cv2.arrowedLine(image, first, second, color, 3, tipLength=0.25)
        for point in pixels:
            cv2.circle(image, point, 6, color, 2)
    if plot is not None:
        impact = tuple(int(round(value)) for value in np.mean(plot["impact_pixels"], axis=0))
        cv2.drawMarker(image, impact, (0, 0, 255), cv2.MARKER_CROSS, 24, 3)
        lines = (
            f"{row['attempt_key']}  bounce f{row['bounce_frame']:g}",
            f"{row['landing_region']}  line {row['line_distance_m']:.2f} m  {row['tournament_week']}",
            f"speed post/pre {row['post_pre_speed_ratio']:.3f}  direction {row['direction_change_deg']:.1f} deg",
        )
    else:
        lines = (
            f"{row['attempt_key']}  bounce f{row['bounce_frame']:g}",
            f"{row['tournament_week']}  model-free measurement abstained",
            str(row["reason"]),
        )
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2
    widths = [cv2.getTextSize(text, font, scale, thickness)[0][0] for text in lines]
    backdrop = image.copy()
    cv2.rectangle(backdrop, (16, 12), (42 + max(widths), 105), (0, 0, 0), -1)
    image = cv2.addWeighted(backdrop, 0.72, image, 0.28, 0.0)
    for index, text in enumerate(lines):
        origin = (28, 39 + 29 * index)
        cv2.putText(image, text, origin, font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    if image.shape[0] != 1080 or image.shape[1] != 1920:
        raise ValueError(
            f"native diagnostic is {image.shape[1]}x{image.shape[0]}, expected 1920x1080"
        )
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 94])


def run_measurement(
    *, preparation_root: Path, output_root: Path, data_root: Path, repo_root: Path
) -> dict[str, Any]:
    from cv.experiments.connected_shooting import cohort

    labels_root = repo_root / "cv/validation/labels/s6_agent_inputs_v1"
    discovered = cohort.discover(labels_root)
    observed_counts = {
        surface: len([attempt for attempt in discovered if attempt.surface == surface])
        for surface in EXPECTED_ATTEMPTS
    }
    grass_keys = {attempt.key for attempt in discovered if attempt.surface == "grass"}
    if observed_counts != EXPECTED_ATTEMPTS or grass_keys != set(GRASS_ATTEMPTS):
        raise ValueError(
            "regional-v2 requires the frozen 14 grass / 51 hard / 24 clay cohort; "
            f"observed counts={observed_counts}, grass={sorted(grass_keys)}"
        )
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    plots: list[
        tuple[
            dict[str, Any],
            dict[str, Any] | None,
            tuple[Path, bool] | None,
            dict[int, dict[str, Any]],
            Path | None,
        ]
    ] = []
    inputs: list[Path] = []
    eligible: dict[str, int] = defaultdict(int)
    for attempt in discovered:
        input_root = preparation_root / attempt.key / "inputs"
        packet_path, cameras_path = input_root / "packet.json", input_root / "cameras.json"
        if not packet_path.is_file() or not cameras_path.is_file():
            rejected.append({"attempt_key": attempt.key, "reason": "missing_preparation"})
            continue
        packet = json.loads(packet_path.read_text())["attempts"][0]
        cameras_document = json.loads(cameras_path.read_text())
        labels = {
            int(row["frame"]): row
            for row in packet["owner_ball_labels"]
            if row.get("status") == "visible"
            and row.get("x1080") is not None
            and row.get("y1080") is not None
        }
        cameras = {
            int(row["frame"]): np.asarray(row["P"], dtype=float)
            for row in cameras_document["cameras"]
            if row.get("status") == "supported" and row.get("P") is not None
        }
        events = sorted(packet["events"], key=lambda row: float(row["frame"]))
        inputs.extend([packet_path, cameras_path])
        video = _video_path(data_root, packet["match_id"], packet["clip"])
        for index, event in enumerate(events):
            if _event_kind(event) != "bounce":
                continue
            eligible[attempt.surface] += 1
            row, reason, plot = measure_bounce(
                key=attempt.key,
                match_id=packet["match_id"],
                surface=attempt.surface,
                events=events,
                index=index,
                labels=labels,
                cameras=cameras,
            )
            overlay_candidates = sorted(input_root.glob("transport_overlays/frame_*.jpg"))
            fallback_image = (
                min(
                    overlay_candidates,
                    key=lambda path: abs(
                        float(path.stem.removeprefix("frame_")) - float(event["frame"])
                    ),
                )
                if overlay_candidates
                else None
            )
            if row is None:
                rejection = {
                    "attempt_key": attempt.key,
                    "match_id": packet["match_id"],
                    "surface": attempt.surface,
                    "bounce_frame": float(event["frame"]),
                    "tournament_week": tournament_week(packet["match_id"]),
                    "reason": reason,
                }
                rejected.append(rejection)
                if attempt.surface == "grass":
                    plots.append((rejection, None, video, labels, fallback_image))
            else:
                rows.append(row)
                if attempt.surface == "grass" and plot is not None:
                    plots.append((row, plot, video, labels, fallback_image))
    profiles = fit_regional_v2(rows)
    _validate_frozen_profiles(profiles)
    quality_rows = [row for row in rows if _usable_measurement(row)]
    payload = {
        "schema": "tennis.grass-regional-v2-model-free.v1",
        "evaluation_only": True,
        "automatic_inference_eligible": False,
        "opened_development_data": True,
        "method": {
            "motion": "linear native leading-front pixel velocity on each side; native_pts_seconds only",
            "direction_change": "angle between pre- and post-bounce image velocity vectors",
            "contact_time": "interpolated native PTS at labeled bounce and next labeled contact",
            "landing": "mean of independently extrapolated pre/post pixels, camera ray at ball-centre height",
            "side_pictures": SIDE_PICTURES,
            "minimum_side_pictures": MIN_SIDE_PICTURES,
            "maximum_side_span_frames": MAX_SIDE_SPAN_FRAMES,
            "dynamics_model": None,
            "spin_assumption": None,
            "quality_gate": {
                "impact_ray_disagreement_m_max": 0.75,
                "side_linear_rms_px_max": 6.0,
                "post_pre_speed_ratio": [0.2, 5.0],
            },
        },
        "denominator": {
            "attempts": dict(
                (surface, observed_counts[surface]) for surface in ("grass", "hard", "clay")
            ),
            "eligible_labeled_bounces": dict(eligible),
            "measured_bounces": dict(
                (surface, len([row for row in rows if row["surface"] == surface]))
                for surface in ("grass", "hard", "clay")
            ),
            "quality_gated_bounces": dict(
                (
                    surface,
                    len([row for row in quality_rows if row["surface"] == surface]),
                )
                for surface in ("grass", "hard", "clay")
            ),
            "rejected_bounces": len([row for row in rejected if "bounce_frame" in row]),
        },
        "summary_by_surface": {
            surface: _summary(
                [row for row in quality_rows if row["surface"] == surface],
                ("landing_region", "tournament_week", "on_line"),
            )
            for surface in ("grass", "hard", "clay")
        },
        "raw_summary_by_surface": {
            surface: _summary(
                [row for row in rows if row["surface"] == surface],
                ("landing_region", "tournament_week", "on_line"),
            )
            for surface in ("grass", "hard", "clay")
        },
        "regional_v2_broadcasts": profiles,
        "measurements": rows,
        "rejected": rejected,
        "provenance": {
            "preparation_root": str(preparation_root.resolve()),
            "labels_root": str(labels_root.resolve()),
            "human_derived_inputs": ["agent ball fronts", "agent event frames", "agent cameras"],
            "input_files": [
                {"path": str(path.resolve()), "sha256": _sha256(path)}
                for path in sorted(set(inputs))
            ],
            "code": {
                **_git_record(repo_root),
                "files": [
                    {
                        "path": str(path.relative_to(repo_root)),
                        "sha256": _sha256(path),
                    }
                    for path in (
                        Path(__file__).resolve(),
                        (repo_root / "physics/surface_model.py").resolve(),
                        (repo_root / "cv/experiments/connected_shooting/cohort.py").resolve(),
                    )
                ],
            },
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "measurements.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    diagnostics = output_root / "native_bounces"
    diagnostics.mkdir(exist_ok=True)
    rendered = []
    for row, plot, video, labels, fallback_image in plots:
        name = f"{row['attempt_key']}__f{str(row['bounce_frame']).replace('.', 'p')}.jpg"
        if video is not None:
            _render_native(diagnostics / name, video, row, labels, plot, fallback_image)
        if (diagnostics / name).is_file():
            rendered.append(name)
    payload["native_diagnostics"] = {
        "requested_grass_bounces": len(plots),
        "rendered": len(rendered),
        "native_dimensions": [1920, 1080],
        "files": rendered,
        "records": [{"file": name, "sha256": _sha256(diagnostics / name)} for name in rendered],
    }
    (output_root / "measurements.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    payload = run_measurement(
        preparation_root=args.preparation_root,
        output_root=args.output_root,
        data_root=args.data_root,
        repo_root=args.repo_root,
    )
    print(
        json.dumps(
            {"denominator": payload["denominator"], "profiles": payload["regional_v2_broadcasts"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
