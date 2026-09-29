"""Automatic court calibration from the full painted-line topology.

This is an experimental peer to :mod:`cv.pipeline.court`. It merges Hough fragments into
distinct image lines, constructs candidate intersection graphs, matches those graphs to the
known tennis-court incidence pattern, and refits the homography from every matched vertex.
"""

from __future__ import annotations

import heapq
import itertools
import os
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from cv.pipeline import court
from cv.pipeline import resolution as res
from cv.pipeline.court_geometry_gate import assess_court_homography

HORIZONTAL_WORLD = (0.0, 5.485, 18.285, 23.77)
VERTICAL_WORLD = (0.0, 1.37, 9.60, 10.97)
MODEL_SEGMENTS = (
    ((0.0, 0.0), (court.COURT_W, 0.0)),
    ((0.0, court.COURT_L), (court.COURT_W, court.COURT_L)),
    ((0.0, 0.0), (0.0, court.COURT_L)),
    ((court.COURT_W, 0.0), (court.COURT_W, court.COURT_L)),
    ((court.SW, 0.0), (court.SW, court.COURT_L)),
    ((court.COURT_W - court.SW, 0.0), (court.COURT_W - court.SW, court.COURT_L)),
    ((court.SW, 5.485), (court.COURT_W - court.SW, 5.485)),
    ((court.SW, 18.285), (court.COURT_W - court.SW, 18.285)),
    ((court.COURT_W / 2.0, 5.485), (court.COURT_W / 2.0, 18.285)),
)
LANDMARKS = tuple(itertools.product(VERTICAL_WORLD, HORIZONTAL_WORLD))


@dataclass(frozen=True)
class LineHypothesis:
    line: np.ndarray
    length: float
    reference: float
    angle_degrees: float


@dataclass(frozen=True)
class CourtTopologyHypothesis:
    homography: np.ndarray
    score: float


@dataclass(frozen=True)
class CourtTopologySolution:
    homography: np.ndarray
    source: str
    topology_score: float
    surface_score: float
    player_score: float | None = None
    refinement_evidence: dict[str, object] | None = None
    proposal_mask: str = "standard"
    # Which acceptance witness admitted this candidate, and the illumination-split surface
    # score when one was measured. See ``ACCEPTANCE_ROUTES``.
    acceptance_route: str = "strict_surface"
    illumination_surface_score: float | None = None


# The painted-edge refinement pushes each identified line onto one of its two visible edges.
# ``itf_outside_edge`` is the documented ``itf_outside_edge_v1`` convention: court dimensions
# are measured to the outside of the paint, so a line is pushed away from the net (baselines
# and service lines) or away from the court centre (sidelines). ``image_lower_edge`` pushes
# every horizontal line towards the bottom of the frame instead, which is what this module
# has always done; the owner landmark truth on both cohorts is clicked in that convention, so
# it stays the default until the truth is re-clicked. See WK2_REPORT.md for the measurement.
# ``paint_centre`` refines each line to the sub-pixel centre of its own paint instead of to
# an edge, which is the convention the landmark truth is closest to.
EDGE_CONVENTIONS = ("itf_outside_edge", "image_lower_edge", "paint_centre")
DEFAULT_EDGE_CONVENTION = "image_lower_edge"
EDGE_CONVENTION_ENVIRONMENT_VARIABLE = "TENNIS_COURT_EDGE_CONVENTION"

VISIBLE_EDGE_WORLD_LINES = (
    ("near_baseline", (0.0, 0.0), (court.COURT_W, 0.0), "near_horizontal"),
    ("near_service", (court.SW, 5.485), (court.COURT_W - court.SW, 5.485), "near_horizontal"),
    ("far_service", (court.SW, 18.285), (court.COURT_W - court.SW, 18.285), "far_horizontal"),
    ("far_baseline", (0.0, court.COURT_L), (court.COURT_W, court.COURT_L), "far_horizontal"),
    ("left_doubles", (0.0, 0.0), (0.0, court.COURT_L), "left_vertical"),
    ("right_doubles", (court.COURT_W, 0.0), (court.COURT_W, court.COURT_L), "right_vertical"),
)
LINE_ORIENTATIONS = ("near_horizontal", "far_horizontal", "left_vertical", "right_vertical")

# ITF painted-line widths: a baseline may be up to 100 mm, every other line is 50 mm.
BASELINE_PAINT_WIDTH_M = 0.10
LINE_PAINT_WIDTH_M = 0.05
PAINT_HALF_WIDTH_M = {
    "y": {
        0.0: BASELINE_PAINT_WIDTH_M / 2.0,
        5.485: LINE_PAINT_WIDTH_M / 2.0,
        18.285: LINE_PAINT_WIDTH_M / 2.0,
        court.COURT_L: BASELINE_PAINT_WIDTH_M / 2.0,
    },
    "x": {
        0.0: LINE_PAINT_WIDTH_M / 2.0,
        court.SW: LINE_PAINT_WIDTH_M / 2.0,
        court.COURT_W - court.SW: LINE_PAINT_WIDTH_M / 2.0,
        court.COURT_W: LINE_PAINT_WIDTH_M / 2.0,
    },
}
# Which way a line is pushed off the paint centre, in world units, per convention. The far
# half of the court is +y and the bottom of a broadcast frame is -y, which is the only place
# the two conventions differ.
_VERTICAL_PUSH = {
    0.0: -1.0,
    court.SW: -1.0,
    court.COURT_W - court.SW: 1.0,
    court.COURT_W: 1.0,
}
LINE_PUSH_DIRECTION = {
    "itf_outside_edge": {
        "y": {0.0: -1.0, 5.485: -1.0, 18.285: 1.0, court.COURT_L: 1.0},
        "x": _VERTICAL_PUSH,
    },
    "image_lower_edge": {
        "y": {0.0: -1.0, 5.485: -1.0, 18.285: -1.0, court.COURT_L: -1.0},
        "x": _VERTICAL_PUSH,
    },
    "paint_centre": {"y": {}, "x": {}},
}


def resolve_edge_convention(edge_convention: str | None) -> str:
    """Pick the painted-edge convention from the argument, the environment, then the default."""
    if edge_convention is None:
        edge_convention = (
            os.environ.get(EDGE_CONVENTION_ENVIRONMENT_VARIABLE) or DEFAULT_EDGE_CONVENTION
        )
    if edge_convention not in EDGE_CONVENTIONS:
        raise ValueError(
            f"unknown edge convention {edge_convention!r}, expected one of {EDGE_CONVENTIONS}"
        )
    return edge_convention


def _paint_centre_offset(axis: str, value: float, edge_convention: str) -> float:
    """How far the paint centre sits from a model line, along ``axis``, in metres."""
    push = LINE_PUSH_DIRECTION[edge_convention][axis].get(value)
    if push is None:
        return 0.0
    return -push * PAINT_HALF_WIDTH_M[axis][value]


def paint_centre_landmarks(edge_convention: str) -> tuple[tuple[float, float], ...]:
    """``LANDMARKS`` moved onto the centre of the paint that carries them."""
    return tuple(
        (
            court_x + _paint_centre_offset("x", court_x, edge_convention),
            court_y + _paint_centre_offset("y", court_y, edge_convention),
        )
        for court_x, court_y in LANDMARKS
    )


def paint_centre_segments(
    edge_convention: str,
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    """``MODEL_SEGMENTS`` moved onto the centre of the paint it is drawn with.

    A segment is offset along whichever axis is constant along it. The centre service line
    is painted symmetrically about its measured position and is left alone.
    """
    segments = []
    for start, end in MODEL_SEGMENTS:
        if start[1] == end[1]:
            offset = (0.0, _paint_centre_offset("y", start[1], edge_convention))
        elif start[0] == end[0]:
            offset = (_paint_centre_offset("x", start[0], edge_convention), 0.0)
        else:
            offset = (0.0, 0.0)
        segments.append(
            (
                (start[0] + offset[0], start[1] + offset[1]),
                (end[0] + offset[0], end[1] + offset[1]),
            )
        )
    return tuple(segments)


def edge_convention_world_transform(
    edge_convention: str | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """World-plane map from the solver's paint-centre frame to a declared edge convention.

    The topology solve scores a candidate by the distance from each projected model line to
    the painted-line mask, so it lands on the centre of the paint. Every label set and
    artifact in this repository declares ``itf_outside_edge_v1``: court dimensions measured
    to the outside of the paint. The two differ by half a published ITF line width, which is
    5 cm at a baseline and 2.5 cm elsewhere - about six native pixels at a near baseline.

    The correction uses only published line widths, never a measured residual against any
    label. It is fitted through the eight model corners, so the returned evidence carries the
    residual of representing a per-line offset by one projective map; a piecewise offset is
    not exactly projective, and that error must be reported rather than assumed away.
    """
    edge_convention = resolve_edge_convention(edge_convention)
    nominal, painted = [], []
    for court_y in (0.0, 5.485, 18.285, court.COURT_L):
        for court_x in (0.0, court.COURT_W):
            nominal.append((court_x, court_y))
            painted.append(
                (
                    court_x + _paint_centre_offset("x", court_x, edge_convention),
                    court_y + _paint_centre_offset("y", court_y, edge_convention),
                )
            )
    transform, _ = cv2.findHomography(
        np.asarray(nominal, dtype=np.float32),
        np.asarray(painted, dtype=np.float32),
        0,
    )
    if transform is None:
        raise ValueError("degenerate edge-convention world transform")
    transform = np.asarray(transform, dtype=float)
    transform = transform / transform[2, 2]
    mapped = cv2.perspectiveTransform(
        np.asarray(nominal, dtype=np.float32).reshape(1, -1, 2), transform
    )[0]
    residuals = np.linalg.norm(mapped - np.asarray(painted, dtype=float), axis=1)
    return transform, {
        "edge_convention": edge_convention,
        "corner_residual_median_m": float(np.median(residuals)),
        "corner_residual_maximum_m": float(residuals.max()),
    }


def to_edge_convention(
    homography: np.ndarray,
    edge_convention: str | None = None,
) -> np.ndarray:
    """Re-express an image-to-court homography in a declared painted-edge convention.

    ``homography`` maps image pixels to the solver's paint-centre court frame; the result
    maps the same pixels to the declared convention's court frame.
    """
    transform, _ = edge_convention_world_transform(edge_convention)
    return transform @ np.asarray(homography, dtype=float)


TOPOLOGY_SCORE_CONVENTIONS = ("nominal", "paint_inset")
# Which convention the refinement's acceptance witness scores the candidate under. "nominal"
# assumes the candidate still puts model lines on the paint, so an edge-refined candidate is
# charged half a line width for being correct and is almost always rejected; that is what
# production has been doing. "paint_inset" scores it under its own edge convention.
DEFAULT_REFINEMENT_WITNESS = "nominal"
REFINEMENT_WITNESS_ENVIRONMENT_VARIABLE = "TENNIS_COURT_REFINEMENT_WITNESS"


def resolve_refinement_witness(witness: str | None) -> str:
    """Pick the refinement acceptance witness from the argument, environment, then default."""
    if witness is None:
        witness = (
            os.environ.get(REFINEMENT_WITNESS_ENVIRONMENT_VARIABLE) or DEFAULT_REFINEMENT_WITNESS
        )
    if witness not in TOPOLOGY_SCORE_CONVENTIONS:
        raise ValueError(
            f"unknown refinement witness {witness!r}, expected one of {TOPOLOGY_SCORE_CONVENTIONS}"
        )
    return witness


SOLVE_SCALES = ("native", "canonical_540")
# The solve runs at 960x540 by default only because that is what the week-one landmark
# cohorts were scored at and native is 0.02 to 0.05 px540 worse on their median; native is
# better on abstention and on precision once the label set's global court offset is removed.
# WK2_REPORT.md has both numbers.
DEFAULT_SOLVE_SCALE = "canonical_540"
SOLVE_SCALE_ENVIRONMENT_VARIABLE = "TENNIS_COURT_SOLVE_SCALE"


def _pixel_scale(width: int) -> float:
    """Pixels per 960-wide canonical pixel.

    Every geometric tolerance in this module is written in canonical 960x540 pixels and
    multiplied by this factor, so a solve at 1920x1080 applies the same tolerances in
    metres of court and reproduces the 540 behaviour exactly when width is 960.
    """
    return float(width) / float(res.CANONICAL_SIZE.width)


def resolve_solve_scale(solve_scale: str | None) -> str:
    """Pick the solve scale from the argument, the environment, then the default."""
    if solve_scale is None:
        solve_scale = os.environ.get(SOLVE_SCALE_ENVIRONMENT_VARIABLE) or DEFAULT_SOLVE_SCALE
    if solve_scale not in SOLVE_SCALES:
        raise ValueError(f"unknown solve scale {solve_scale!r}, expected one of {SOLVE_SCALES}")
    return solve_scale


def relaxed_chroma_line_mask(
    image: np.ndarray,
    *,
    top_fraction: float = 0.12,
) -> np.ndarray:
    """Find bright painted lines whose JPEG edges inherit saturated court color.

    This is a candidate-generation arm for clay and strongly compressed footage. The
    strict topology score remains the acceptance gate, so the broader mask cannot by
    itself certify a calibration.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel_size = max(9, int(round(15 * image.shape[1] / 960)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    top_hat = cv2.morphologyEx(
        gray,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)),
    )
    value = hsv[..., 2]
    mask = ((top_hat > 22) & (hsv[..., 1] < 190) & (value > 105)).astype(np.uint8) * 255
    mask[~court.line_mask_observation_mask(mask.shape, top_fraction)] = 0
    return mask


def court_line_masks(image: np.ndarray) -> tuple[tuple[str, np.ndarray], ...]:
    """Return independent court-line proposal masks in preference order."""
    return (
        ("standard", court.line_mask(image, top_fraction=0.12)),
        ("relaxed_chroma", relaxed_chroma_line_mask(image, top_fraction=0.12)),
    )


REGISTRATION_MASK_POLICIES = ("off", "court_observation", "court_observation_fallback")
INTERPOLATED_CAMERA_POLICIES = ("off", "qualify_retry")
REGISTRATION_MASK_TOP_FRACTION = 0.22


def register_static_scene(
    target_image: np.ndarray,
    reference_image: np.ndarray,
    *,
    minimum_inliers: int = 40,
    minimum_inlier_ratio: float = 0.35,
    court_observation_mask: bool = False,
) -> tuple[np.ndarray, dict[str, float | int]] | None:
    """Estimate target-image to reference-image motion from static-scene features."""
    detector = cv2.SIFT_create(nfeatures=4000)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    def feature_mask(image: np.ndarray) -> np.ndarray | None:
        if not court_observation_mask:
            return None
        return (
            court.line_mask_observation_mask(
                image.shape, top_fraction=REGISTRATION_MASK_TOP_FRACTION
            ).astype(np.uint8)
            * 255
        )

    target_keys, target_descriptors = detector.detectAndCompute(
        target_gray, feature_mask(target_image)
    )
    reference_keys, reference_descriptors = detector.detectAndCompute(
        reference_gray, feature_mask(reference_image)
    )
    if target_descriptors is None or reference_descriptors is None:
        return None
    pairs = cv2.BFMatcher().knnMatch(target_descriptors, reference_descriptors, k=2)
    matches = [
        pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.70 * pair[1].distance
    ]
    if len(matches) < minimum_inliers:
        return None
    target_points = np.float32([target_keys[row.queryIdx].pt for row in matches])
    reference_points = np.float32([reference_keys[row.trainIdx].pt for row in matches])
    transform, inlier_mask = cv2.findHomography(
        target_points,
        reference_points,
        cv2.RANSAC,
        3.0,
    )
    if transform is None or inlier_mask is None:
        return None
    inliers = int(inlier_mask.sum())
    inlier_ratio = inliers / len(matches)
    if inliers < minimum_inliers or inlier_ratio < minimum_inlier_ratio:
        return None
    return transform, {
        "matches": len(matches),
        "inliers": inliers,
        "inlier_ratio": float(inlier_ratio),
    }


# Support required when only the broader clay/compression mask carries the painted lines.
# It is the same bar the line-model recovery in ``court_topology_runner`` already applies to
# that mask, so a relaxed acceptance is never made on weaker evidence than the module
# already accepts elsewhere.
MINIMUM_RELAXED_TARGET_SCORE = 0.92


def transfer_court_homography(
    target_image: np.ndarray,
    reference_image: np.ndarray,
    reference_homography: np.ndarray,
    *,
    minimum_target_score: float = 0.80,
    minimum_relaxed_target_score: float = MINIMUM_RELAXED_TARGET_SCORE,
    registration_mask: str = "off",
) -> tuple[np.ndarray, dict]:
    """Transfer an anchor using source features and unchanged target-paint checks.

    Screen-fixed graphics must not be confused with physical scene motion. The named
    court observation mask excludes the existing graphics/crowd margins from SIFT
    extraction. ``court_observation`` applies it to every registration;
    ``court_observation_fallback`` retries only when the original transfer is rejected.
    Both use exactly the same feature-match, geometry and painted-line thresholds.
    """
    if registration_mask not in REGISTRATION_MASK_POLICIES:
        raise ValueError(f"unknown court registration mask {registration_mask!r}")
    attempts = {
        "off": (False,),
        "court_observation": (True,),
        "court_observation_fallback": (False, True),
    }[registration_mask]
    original_failure = None
    for masked in attempts:
        try:
            registered = register_static_scene(
                target_image,
                reference_image,
                **({"court_observation_mask": True} if masked else {}),
            )
            candidate, evidence = _qualify_court_transfer(
                target_image,
                reference_homography,
                registered,
                minimum_target_score,
                minimum_relaxed_target_score,
            )
        except (ValueError, np.linalg.LinAlgError, cv2.error) as exc:
            if not masked and registration_mask == "court_observation_fallback":
                original_failure = str(exc)
                continue
            raise
        if masked:
            evidence = {
                **evidence,
                "registration_feature_mask": "court_observation_v1",
                "registration_mask_top_fraction": REGISTRATION_MASK_TOP_FRACTION,
                "original_transfer_failure": original_failure,
            }
        return candidate, evidence
    raise AssertionError("registration attempts must return or raise")


def _qualify_court_transfer(
    target_image: np.ndarray,
    reference_homography: np.ndarray,
    registered: tuple[np.ndarray, dict[str, float | int]] | None,
    minimum_target_score: float,
    minimum_relaxed_target_score: float,
) -> tuple[np.ndarray, dict]:
    if registered is None:
        raise ValueError("static-scene registration failed")
    transform, evidence = registered
    candidate = reference_homography @ transform
    witness = assess_court_image_support(
        target_image,
        candidate,
        minimum_target_score=minimum_target_score,
        minimum_relaxed_target_score=minimum_relaxed_target_score,
    )
    if not witness["accepted"]:
        raise ValueError(witness["reason"])
    return candidate, {**evidence, **{k: v for k, v in witness.items() if k != "accepted"}}


def assess_court_image_support(
    target_image: np.ndarray,
    candidate: np.ndarray,
    *,
    minimum_target_score: float = 0.80,
    minimum_relaxed_target_score: float = MINIMUM_RELAXED_TARGET_SCORE,
) -> dict:
    """Measure a candidate on its own picture without inventing feature evidence.

    Registered samples and interpolations share the same geometry and painted-line
    thresholds. Rejected paint scores remain available for a source-evidence receipt.
    """
    height, width = target_image.shape[:2]
    if not _standard_broadcast_view(candidate, width, height):
        return {
            "accepted": False,
            "reason": "registered court geometry is not a standard broadcast view",
        }
    observation_mask = court.line_mask_observation_mask(target_image.shape, top_fraction=0.12)
    support = {
        name: float(topology_score(candidate, mask, observation_mask=observation_mask))
        for name, mask in court_line_masks(target_image)
    }
    thresholds = {"standard": minimum_target_score, "relaxed_chroma": minimum_relaxed_target_score}
    supported = {
        name: score for name, score in support.items() if score >= thresholds.get(name, 1.0)
    }
    if not supported:
        return {
            "accepted": False,
            "reason": (
                "registered court lacks target support "
                f"(standard={support['standard']:.3f}, "
                f"relaxed_chroma={support['relaxed_chroma']:.3f})"
            ),
            "target_support": support,
        }
    support_mask = max(supported, key=lambda name: supported[name] - thresholds[name])
    return {
        "accepted": True,
        "target_topology_score": support[support_mask],
        "target_support": support,
        "target_support_mask": support_mask,
    }


def _line_from_points(points: np.ndarray) -> np.ndarray:
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    line = np.asarray([vy, -vx, vx * y0 - vy * x0], dtype=float)
    norm = np.hypot(line[0], line[1])
    return line / norm


def _segment_line(segment: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = segment
    line = np.cross([x1, y1, 1.0], [x2, y2, 1.0]).astype(float)
    return line / np.hypot(line[0], line[1])


def _angle(line: np.ndarray) -> float:
    direction = np.asarray([-line[1], line[0]])
    value = abs(np.degrees(np.arctan2(direction[1], direction[0]))) % 180.0
    return min(value, 180.0 - value)


def _reference(line: np.ndarray, family: str, width: int, height: int) -> float:
    if family == "horizontal":
        x = width / 2.0
        return float(-(line[0] * x + line[2]) / line[1])
    y = 0.58 * height
    return float(-(line[1] * y + line[2]) / line[0])


def _merge_segments(
    segments: list[np.ndarray],
    family: str,
    width: int,
    height: int,
    *,
    merge_pixels: float,
    merge_angle_degrees: float = 4.5,
) -> list[LineHypothesis]:
    rows = []
    for segment in segments:
        line = _segment_line(segment)
        rows.append(
            {
                "segment": segment,
                "line": line,
                "reference": _reference(line, family, width, height),
                "angle": _angle(line),
                "length": float(np.hypot(segment[2] - segment[0], segment[3] - segment[1])),
            }
        )
    rows.sort(key=lambda row: row["reference"])
    clusters: list[list[dict]] = []
    for row in rows:
        target = None
        for cluster in reversed(clusters[-3:]):
            weight = sum(item["length"] for item in cluster)
            reference = sum(item["reference"] * item["length"] for item in cluster) / weight
            angle = sum(item["angle"] * item["length"] for item in cluster) / weight
            if (
                abs(row["reference"] - reference) <= merge_pixels
                and abs(row["angle"] - angle) <= merge_angle_degrees
            ):
                target = cluster
                break
        if target is None:
            clusters.append([row])
        else:
            target.append(row)
    hypotheses = []
    for cluster in clusters:
        points = np.concatenate(
            [item["segment"].reshape(2, 2) for item in cluster],
            axis=0,
        )
        line = _line_from_points(points)
        hypotheses.append(
            LineHypothesis(
                line=line,
                length=sum(item["length"] for item in cluster),
                reference=_reference(line, family, width, height),
                angle_degrees=_angle(line),
            )
        )
    return sorted(hypotheses, key=lambda row: row.length, reverse=True)


def detect_line_families(
    image: np.ndarray,
    *,
    top_fraction: float = 0.12,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[LineHypothesis], list[LineHypothesis]]:
    mask = court.line_mask(image, top_fraction=top_fraction) if mask is None else mask
    height, width = mask.shape
    segments = cv2.HoughLinesP(
        mask,
        1,
        np.pi / 360.0,
        threshold=max(35, round(0.05 * width)),
        minLineLength=max(45, round(0.055 * width)),
        maxLineGap=max(12, round(0.025 * width)),
    )
    horizontal = []
    longitudinal = []
    for segment in segments.reshape(-1, 4) if segments is not None else []:
        angle = abs(np.degrees(np.arctan2(segment[3] - segment[1], segment[2] - segment[0])))
        angle = min(angle % 180.0, 180.0 - (angle % 180.0))
        if angle <= 24.0:
            horizontal.append(segment.astype(float))
        elif angle >= 32.0:
            longitudinal.append(segment.astype(float))
    scale = width / 960.0
    return (
        mask,
        _merge_segments(horizontal, "horizontal", width, height, merge_pixels=6.0 * scale),
        _merge_segments(longitudinal, "vertical", width, height, merge_pixels=7.0 * scale),
    )


def _spatially_balanced_hypotheses(
    hypotheses: list[LineHypothesis],
    extent: int,
    *,
    limit: int = 20,
    bins: int = 16,
) -> list[LineHypothesis]:
    """Retain strong lines without letting one advertising band consume the budget."""
    selected: list[LineHypothesis] = []
    selected_ids: set[int] = set()

    def add(row: LineHypothesis) -> None:
        row_id = id(row)
        if row_id not in selected_ids and len(selected) < limit:
            selected.append(row)
            selected_ids.add(row_id)

    for row in hypotheses[:4]:
        add(row)
    binned = []
    for bin_index in range(bins):
        low = extent * bin_index / bins
        high = extent * (bin_index + 1) / bins
        binned.append([row for row in hypotheses if low <= row.reference < high])
    for rows in binned:
        if rows:
            add(rows[0])
    for rows in binned:
        for row in rows[1:2]:
            add(row)
    for row in hypotheses:
        add(row)
    return selected


def _intersection(first: np.ndarray, second: np.ndarray) -> tuple[float, float] | None:
    point = np.cross(first, second)
    if abs(point[2]) < 1e-8:
        return None
    return float(point[0] / point[2]), float(point[1] / point[2])


def _project_court_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        points.reshape(1, -1, 2).astype(np.float32), np.linalg.inv(homography)
    )[0]


def _standard_broadcast_view(
    homography: np.ndarray,
    width: int,
    height: int,
) -> bool:
    geometry = assess_court_homography(homography, width, height)
    if not geometry["valid"]:
        return False
    centers = geometry["line_centers_y"]
    spans = geometry["line_spans_px"]
    image_lines = [
        _project_court_points(
            homography,
            np.asarray([(0.0, court_y), (court.COURT_W, court_y)]),
        )
        for court_y in (0.0, court.NET_Y, court.COURT_L)
    ]
    return (
        # Vertical framing can translate a valid court within the picture. Perspective,
        # line order and visible court coverage carry the view evidence instead.
        spans["near_baseline"] > spans["net"] > spans["far_baseline"]
        and centers["near_baseline"] - centers["far_baseline"] >= 0.40 * height
        and 0.45 * width <= spans["near_baseline"] <= 0.95 * width
        and 0.22 * width <= spans["net"] <= 0.70 * width
        and 0.12 * width <= spans["far_baseline"] <= 0.60 * width
        and all(abs(line[1, 1] - line[0, 1]) <= 0.05 * height for line in image_lines)
        and all(-0.10 * width <= line[:, 0].min() for line in image_lines)
        and all(line[:, 0].max() <= 1.10 * width for line in image_lines)
    )


def _distance_samples(mask: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3)


def _vertex_score(
    homography: np.ndarray,
    distance: np.ndarray,
    *,
    convention: str = "nominal",
    edge_convention: str | None = None,
    observation_mask: np.ndarray | None = None,
) -> float:
    height, width = distance.shape
    scale = _pixel_scale(width)
    model = (
        LANDMARKS
        if convention == "nominal"
        else paint_centre_landmarks(resolve_edge_convention(edge_convention))
    )
    landmarks = _project_court_points(homography, np.asarray(model))
    inside = (
        (landmarks[:, 0] >= 0)
        & (landmarks[:, 0] < width)
        & (landmarks[:, 1] >= 0)
        & (landmarks[:, 1] < height)
    )
    if observation_mask is not None:
        visible = np.flatnonzero(inside)
        pixels = np.floor(landmarks[visible]).astype(int)
        inside[visible] &= observation_mask[pixels[:, 1], pixels[:, 0]]
    if inside.mean() < 0.5:
        return 0.0
    pixels = np.floor(landmarks[inside]).astype(int)
    residual = np.minimum(distance[pixels[:, 1], pixels[:, 0]], 100.0 * scale)
    return float(np.mean(np.exp(-0.5 * (residual / (5.0 * scale)) ** 2)))


def topology_score(
    homography: np.ndarray,
    mask: np.ndarray,
    *,
    distance: np.ndarray | None = None,
    convention: str = "nominal",
    edge_convention: str | None = None,
    observation_mask: np.ndarray | None = None,
) -> float:
    """Score how well the painted-line mask supports a homography.

    An explicit observation mask excludes pixels deliberately erased by the extractor,
    such as graphics. Missing paint in retained pixels remains negative evidence. The
    existing minimum visible line, segment and vertex coverage still applies.

    ``convention="nominal"`` assumes the homography puts the model lines on the paint, which
    is what a line-centre topology fit produces. ``convention="paint_inset"`` assumes it puts
    them on a visible edge of the paint under ``edge_convention`` and therefore scores against
    paint centres inset by half a line width. Scoring an edge-refined candidate as nominal
    penalises it by half a line width for doing exactly what it was asked to do, which is why
    the painted-edge refinement used to be rejected on almost every frame.
    """
    if convention not in TOPOLOGY_SCORE_CONVENTIONS:
        raise ValueError(f"unknown topology score convention {convention!r}")
    height, width = mask.shape
    if observation_mask is not None and (
        observation_mask.shape != mask.shape or observation_mask.dtype != np.bool_
    ):
        raise ValueError("observation_mask must be boolean and match the line mask")
    scale = _pixel_scale(width)
    distance = _distance_samples(mask) if distance is None else distance
    line_scores = []
    segments = (
        MODEL_SEGMENTS
        if convention == "nominal"
        else paint_centre_segments(resolve_edge_convention(edge_convention))
    )
    for start, end in segments:
        samples = np.linspace(start, end, 72)
        projected = _project_court_points(homography, samples)
        inside = (
            (projected[:, 0] >= 0)
            & (projected[:, 0] < width)
            & (projected[:, 1] >= 0)
            & (projected[:, 1] < height)
        )
        if observation_mask is not None:
            visible = np.flatnonzero(inside)
            pixels = np.floor(projected[visible]).astype(int)
            inside[visible] &= observation_mask[pixels[:, 1], pixels[:, 0]]
        if inside.mean() < 0.25:
            continue
        pixels = np.floor(projected[inside]).astype(int)
        residual = np.minimum(distance[pixels[:, 1], pixels[:, 0]], 100.0 * scale)
        line_scores.append(float(np.mean(np.exp(-0.5 * (residual / (4.0 * scale)) ** 2))))
    if len(line_scores) < 6:
        return 0.0
    vertex_score = _vertex_score(
        homography,
        distance,
        convention=convention,
        edge_convention=edge_convention,
        observation_mask=observation_mask,
    )
    ordered = sorted(line_scores)
    lower = float(np.mean(ordered[: max(2, len(ordered) // 2)]))
    return 0.50 * float(np.mean(line_scores)) + 0.30 * lower + 0.20 * vertex_score


# The LAB distance a sampled interior patch may sit from the colour it is scored against.
# Both surface witnesses use it, unchanged from the single-median witness that introduced it.
SURFACE_COLOUR_RESIDUAL = 20.0


def _court_interior_lab_samples(image: np.ndarray, homography: np.ndarray) -> np.ndarray | None:
    """Per-patch median LAB colour over the proposed court interior.

    ``None`` when too little of the proposed interior is inside the frame to judge, which
    both surface witnesses report as a score of zero.
    """
    court_points = np.asarray(
        [
            (court_x, court_y)
            for court_y in np.linspace(0.8, court.COURT_L - 0.8, 18)
            for court_x in np.linspace(0.5, court.COURT_W - 0.5, 10)
        ],
        dtype=np.float32,
    )
    projected = _project_court_points(homography, court_points)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(float)
    height, width = lab.shape[:2]
    radius = max(2, int(round(2.0 * _pixel_scale(width))))
    samples = []
    for pixel_x, pixel_y in projected:
        x = int(round(pixel_x))
        y = int(round(pixel_y))
        if not radius <= x < width - radius or not radius <= y < height - radius:
            continue
        patch = lab[y - radius : y + radius + 1, x - radius : x + radius + 1]
        samples.append(np.median(patch.reshape(-1, 3), axis=0))
    if len(samples) < 0.85 * len(court_points):
        return None
    return np.asarray(samples)


def court_surface_consistency(image: np.ndarray, homography: np.ndarray) -> float:
    """Measure whether the proposed court interior is one coherent playing surface."""
    colors = _court_interior_lab_samples(image, homography)
    if colors is None:
        return 0.0
    center = np.median(colors, axis=0)
    residuals = np.linalg.norm(colors - center, axis=1)
    return float(np.mean(residuals < SURFACE_COLOUR_RESIDUAL))


def _two_lightness_clusters(lightness: np.ndarray) -> np.ndarray:
    """Split samples into two lightness clusters.

    The exact two-means split of the sampled lightness, found by scanning every split of the
    sorted values with prefix sums, so it needs no seed, no iteration limit and no threshold
    on L: the split is a deterministic function of the samples alone. Being exact also keeps
    a stray bright sample -- a painted line the interior grid happens to land on -- from
    taking a cluster to itself, which a two-means seeded at the extremes does.
    """
    order = np.argsort(lightness, kind="stable")
    values = np.asarray(lightness, dtype=float)[order]
    count = len(values)
    if count < 2 or values[-1] - values[0] < 1e-9:
        return np.zeros(count, dtype=int)
    total = np.cumsum(values)
    squares = np.cumsum(values**2)
    left = np.arange(1, count)
    left_error = squares[:-1] - total[:-1] ** 2 / left
    right_error = (squares[-1] - squares[:-1]) - (total[-1] - total[:-1]) ** 2 / (count - left)
    split = int(np.argmin(left_error + right_error)) + 1
    assignment = np.empty(count, dtype=int)
    assignment[order[:split]] = 0
    assignment[order[split:]] = 1
    return assignment


def court_surface_illumination_consistency(image: np.ndarray, homography: np.ndarray) -> float:
    """Measure the same surface coherence across one illumination split.

    Identical samples and identical full-LAB residual as ``court_surface_consistency``. The
    only difference is that the samples are first split into two lightness clusters and each
    residual is measured against its own cluster's median colour, so a court cut in two by
    hard sun and shadow is scored as the one surface it is. Lightness is kept inside each
    cluster rather than dropped, so two genuinely different materials at the same chroma are
    still charged for their difference.
    """
    colors = _court_interior_lab_samples(image, homography)
    if colors is None:
        return 0.0
    assignment = _two_lightness_clusters(colors[:, 0])
    residuals = np.zeros(len(colors))
    for index in (0, 1):
        members = assignment == index
        if not members.any():
            continue
        center = np.median(colors[members], axis=0)
        residuals[members] = np.linalg.norm(colors[members] - center, axis=1)
    return float(np.mean(residuals < SURFACE_COLOUR_RESIDUAL))


def player_foot_geometry_score(
    homography: np.ndarray,
    player_feet: np.ndarray,
) -> float:
    """Broad sanity witness for two image-space player foot locations.

    The witness deliberately allows net play and positions behind either baseline. It is
    additive evidence, not a court solver or a hard requirement.
    """
    feet = np.asarray(player_feet, dtype=np.float32).reshape(-1, 2)
    if len(feet) < 2:
        return 0.5
    feet = feet[np.argsort(feet[:, 1])]
    far_xy, near_xy = cv2.perspectiveTransform(feet[[0, -1]].reshape(1, -1, 2), homography)[0]
    checks = (
        -4.0 <= far_xy[0] <= court.COURT_W + 4.0,
        -4.0 <= near_xy[0] <= court.COURT_W + 4.0,
        court.NET_Y - 2.0 <= far_xy[1] <= court.COURT_L + 8.0,
        -8.0 <= near_xy[1] <= court.NET_Y + 2.0,
        far_xy[1] - near_xy[1] >= 4.0,
    )
    return float(np.mean(checks))


# The surface witness barely separates a correct court from a wrong one: sliding a solved
# court eight metres up the frame changes its measured surface consistency by a few points
# on hard courts and not at all on clay, because the surround is the same material. It does
# reject a candidate that lands on the crowd, so it is kept as a coarse guard. What it also
# does is reject a correct court split by hard sun and shadow, where the same paint scores
# 0.996 on topology and 0.606 on surface. Topology is the witness that separates: on the
# controls above, a correct court scores 0.93-1.00 and every displaced or rescaled one at
# most 0.889. A candidate with topology at or above ``strong_topology_score`` is therefore
# accepted on the coarse surface guard alone, and says so in its source.
NO_TOPOLOGY_SOLUTION = "no court topology passed topology and surface witnesses"
STRONG_TOPOLOGY_SCORE = 0.92
STRONG_TOPOLOGY_MINIMUM_SURFACE_SCORE = 0.50

# The case the coarse guard was written for is also the case it still rejects. A grass court
# cut across by hard sun and shadow measures topology 0.99 and surface 0.20-0.40, because
# ``court_surface_consistency`` scores an absolute colour whose L carries the shadow.
# ``illumination_split`` consults ``court_surface_illumination_consistency`` instead, on the
# same samples and the same LAB residual, and only: where the coarse guard already applies
# (topology at or above ``strong_topology_score``), against the same unchanged 0.50 constant,
# and only for a candidate that no arm of the ordinary admission accepted. It is no sharper
# than the guard it stands in for: measured on that same anchor frame, a court displaced 200 px
# scores 0.79 on it and a court rescaled x0.75 scores 0.72, where the single-median witness
# scores them 0.07 and 0.41. Neither witness separates a wrong court from a right one, and the
# current one does not either, so the witnesses that actually separate are unchanged: topology
# at or above ``strong_topology_score``, the painted-edge and far-baseline geometry, and the
# independent net-based camera qualification downstream. Default off.
SURFACE_WITNESS_POLICIES = ("off", "illumination_split")
# ``off`` is the strict route.  Any other policy adds ONE further anchor pass after the
# strict passes have already failed, so enabling it cannot change a frame the strict route
# already solved.  Measured 2026-09-17 on the six fresh broadcasts: Halle 0 -> 79.9% of
# frames and 99.8% of in-play frames supported, four points from abstained to direct, and
# exactly zero change on the other five sources.  Compare against SURFACE_WITNESS_OFF, never
# against the default, or flipping the default silently disables the fallback it enables.
SURFACE_WITNESS_OFF = "off"
# The low-level solver stays strict: flipping THIS default would enable the illumination
# recovery arm at line ~1076 for every direct caller, which admits a 0.40 surface score the
# strict witness rejects.  Only the pipeline entrypoints below default to the split policy,
# and the runner applies it as ONE extra anchor pass after the strict passes have failed.
DEFAULT_SURFACE_WITNESS_POLICY = "off"
PIPELINE_SURFACE_WITNESS_POLICY = "illumination_split"
STRICT_SURFACE_ROUTE = "strict_surface"
COARSE_SURFACE_ROUTE = "strong_topology_coarse_surface"
ILLUMINATION_SPLIT_ROUTE = "strong_topology_illumination_split"
ACCEPTANCE_ROUTES = (STRICT_SURFACE_ROUTE, COARSE_SURFACE_ROUTE, ILLUMINATION_SPLIT_ROUTE)


def resolve_surface_witness_policy(policy: str | None) -> str:
    """Pick the surface-witness policy from the argument, then the default."""
    if policy is None:
        return DEFAULT_SURFACE_WITNESS_POLICY
    if policy not in SURFACE_WITNESS_POLICIES:
        raise ValueError(
            f"unknown surface witness policy {policy!r}, expected one of {SURFACE_WITNESS_POLICIES}"
        )
    return policy


def solve_court_h_topology(
    image: np.ndarray,
    *,
    minimum_topology_score: float = 0.80,
    minimum_surface_score: float = 0.82,
    strong_topology_score: float = STRONG_TOPOLOGY_SCORE,
    strong_topology_minimum_surface_score: float = STRONG_TOPOLOGY_MINIMUM_SURFACE_SCORE,
    player_feet: np.ndarray | None = None,
    minimum_player_score: float = 0.60,
    proposal_pool: int = 48,
    surface_witness_policy: str | None = None,
) -> CourtTopologySolution:
    """Compose strict and expanded proposals under independent acceptance witnesses."""
    surface_witness_policy = resolve_surface_witness_policy(surface_witness_policy)
    arms = (
        ("standard", court.line_mask(image, top_fraction=0.12), False),
        ("standard_balanced", court.line_mask(image, top_fraction=0.12), True),
        ("relaxed_chroma_balanced", relaxed_chroma_line_mask(image), True),
    )
    accepted = []
    # Strong-topology candidates that only the coarse surface guard turned away. They are
    # reconsidered under ``illumination_split`` after every arm of the ordinary admission has
    # failed, so a candidate admitted today keeps its route, its ranking and its homography.
    deferred: list[tuple[str, CourtTopologyHypothesis, float]] = []
    for source, mask, balanced in arms:
        try:
            hypotheses = court_topology_hypotheses(
                image,
                mask=mask,
                spatially_balanced=balanced,
                limit=3,
                **({"proposal_pool": proposal_pool} if proposal_pool != 48 else {}),
            )
        except (ValueError, np.linalg.LinAlgError, cv2.error):
            continue
        for hypothesis in hypotheses:
            if hypothesis.score < minimum_topology_score:
                continue
            surface_score = court_surface_consistency(image, hypothesis.homography)
            strong = (
                hypothesis.score >= strong_topology_score
                and surface_score >= strong_topology_minimum_surface_score
            )
            if surface_score < minimum_surface_score and not strong:
                if hypothesis.score >= strong_topology_score:
                    deferred.append((source, hypothesis, surface_score))
                continue
            player_score = (
                player_foot_geometry_score(hypothesis.homography, player_feet)
                if player_feet is not None
                else None
            )
            if player_score is not None and player_score < minimum_player_score:
                continue
            accepted.append(
                CourtTopologySolution(
                    homography=hypothesis.homography,
                    source=(
                        f"{source}_strong_topology"
                        if surface_score < minimum_surface_score
                        else source
                    ),
                    topology_score=hypothesis.score,
                    surface_score=surface_score,
                    player_score=player_score,
                    proposal_mask="relaxed_chroma"
                    if source.startswith("relaxed_chroma")
                    else "standard",
                    acceptance_route=(
                        COARSE_SURFACE_ROUTE
                        if surface_score < minimum_surface_score
                        else STRICT_SURFACE_ROUTE
                    ),
                )
            )
        if accepted and source == "standard":
            break
    if not accepted and surface_witness_policy == "illumination_split":
        for source, hypothesis, surface_score in deferred:
            illumination_score = court_surface_illumination_consistency(
                image, hypothesis.homography
            )
            if illumination_score < strong_topology_minimum_surface_score:
                continue
            player_score = (
                player_foot_geometry_score(hypothesis.homography, player_feet)
                if player_feet is not None
                else None
            )
            if player_score is not None and player_score < minimum_player_score:
                continue
            accepted.append(
                CourtTopologySolution(
                    homography=hypothesis.homography,
                    source=f"{source}_strong_topology_illumination_split",
                    topology_score=hypothesis.score,
                    surface_score=surface_score,
                    player_score=player_score,
                    proposal_mask="relaxed_chroma"
                    if source.startswith("relaxed_chroma")
                    else "standard",
                    acceptance_route=ILLUMINATION_SPLIT_ROUTE,
                    illumination_surface_score=illumination_score,
                )
            )
    if not accepted:
        raise ValueError(NO_TOPOLOGY_SOLUTION)
    return max(
        accepted,
        key=lambda row: (
            0.70 * row.topology_score
            + 0.25
            * (
                row.surface_score
                if row.illumination_surface_score is None
                else row.illumination_surface_score
            )
            + 0.05 * (row.player_score if row.player_score is not None else 1.0)
        ),
    )


def solve_court_h_topology_native(
    image: np.ndarray,
    *,
    player_feet: np.ndarray | None = None,
    solve_scale: str | None = None,
    edge_convention: str | None = None,
    witness_convention: str | None = None,
    proposal_pool: int = 48,
    surface_witness_policy: str | None = None,
) -> CourtTopologySolution:
    """Solve the court topology and return a homography in the source image's pixels.

    ``solve_scale="native"`` runs the whole solve and the painted-edge refinement on the
    frame as supplied; every geometric tolerance in this module is scaled with the frame
    width, so the search covers the same distance on court either way.
    ``solve_scale="canonical_540"`` reproduces the previous behaviour: resize to 960x540
    with ``INTER_AREA``, solve there, and scale the homography back up.
    """
    solve_scale = resolve_solve_scale(solve_scale)
    source_size = res.FrameSize(image.shape[1], image.shape[0])
    if solve_scale == "native":
        processing_image = image
        processing_size = source_size
        processing_feet = player_feet
    else:
        processing_size = res.CANONICAL_SIZE
        processing_image = cv2.resize(
            image,
            (processing_size.width, processing_size.height),
            interpolation=cv2.INTER_AREA,
        )
        processing_feet = None
        if player_feet is not None:
            processing_feet = np.asarray(player_feet, dtype=float).copy().reshape(-1, 2)
            processing_feet[:, 0] *= processing_size.width / source_size.width
            processing_feet[:, 1] *= processing_size.height / source_size.height
    solution = solve_court_h_topology(
        processing_image,
        player_feet=processing_feet,
        surface_witness_policy=surface_witness_policy,
        **({"proposal_pool": proposal_pool} if proposal_pool != 48 else {}),
    )
    refined_homography, refinement_evidence = refine_visible_painted_edges(
        processing_image,
        solution.homography,
        edge_convention=edge_convention,
        witness_convention=witness_convention,
        witness_mask=solution.proposal_mask,
    )
    refinement_evidence.update(
        proposal_mask=solution.proposal_mask,
        proposal_topology_score=float(solution.topology_score),
        proposal_topology_convention="nominal",
    )
    if proposal_pool != 48:
        refinement_evidence["proposal_pool"] = proposal_pool
    refinement_evidence["solve_scale"] = solve_scale
    refinement_evidence["solve_size"] = processing_size.label
    refinement_evidence["acceptance_route"] = solution.acceptance_route
    illumination_surface_score = solution.illumination_surface_score
    if illumination_surface_score is not None and refinement_evidence["accepted"]:
        # Report both surface scores on the same homography the solution carries.
        illumination_surface_score = court_surface_illumination_consistency(
            processing_image, refined_homography
        )
    return CourtTopologySolution(
        homography=res.image_to_world_homography(
            refined_homography,
            processing_size,
            source_size,
        ),
        source=solution.source,
        topology_score=float(
            refinement_evidence.get("refined_topology_score", solution.topology_score)
            if refinement_evidence["accepted"]
            else solution.topology_score
        ),
        surface_score=float(
            refinement_evidence.get("refined_surface_score", solution.surface_score)
            if refinement_evidence["accepted"]
            else solution.surface_score
        ),
        player_score=solution.player_score,
        refinement_evidence=refinement_evidence,
        proposal_mask=solution.proposal_mask,
        acceptance_route=solution.acceptance_route,
        illumination_surface_score=illumination_surface_score,
    )


def _line_cost(
    observed: LineHypothesis, projected: np.ndarray, family: str, width: int, height: int
) -> float:
    reference = _reference(projected, family, width, height)
    scale = height if family == "horizontal" else width
    angle_delta = abs(observed.angle_degrees - _angle(projected)) / 12.0
    return abs(observed.reference - reference) / (0.04 * scale) + angle_delta


def _projected_line(
    homography: np.ndarray, start: tuple[float, float], end: tuple[float, float]
) -> np.ndarray:
    points = _project_court_points(homography, np.asarray([start, end]))
    return _segment_line(points.ravel())


def _oriented_line_normal(
    line: np.ndarray,
    orientation: str,
    edge_convention: str | None = None,
) -> np.ndarray:
    """Point a line's normal at the visible edge it should be refined onto, in image space.

    Under ``itf_outside_edge`` the normal faces away from the net for the near baseline and
    near service line (image-down in a standard broadcast view) and away from the net for the
    far service line and far baseline too, which is image-up. Under ``image_lower_edge`` every
    horizontal line points image-down. Sidelines always point away from the court centre.
    """
    edge_convention = resolve_edge_convention(edge_convention)
    if orientation not in LINE_ORIENTATIONS:
        raise ValueError(f"unknown line orientation {orientation!r}")
    image_up = orientation == "far_horizontal" and edge_convention == "itf_outside_edge"
    normal = np.asarray(line[:2], dtype=float).copy()
    if orientation in ("near_horizontal", "far_horizontal"):
        if (normal[1] > 0) if image_up else (normal[1] < 0):
            normal *= -1
    elif orientation == "left_vertical" and normal[0] > 0:
        normal *= -1
    elif orientation == "right_vertical" and normal[0] < 0:
        normal *= -1
    return normal


def _painted_edge_profile(
    top_hat: np.ndarray,
    points: np.ndarray,
    normal: np.ndarray,
    *,
    maximum_offset: float,
    offset_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(-maximum_offset, maximum_offset + 0.5 * offset_step, offset_step)
    sample_x = points[:, 0][None, :] + offsets[:, None] * normal[0]
    sample_y = points[:, 1][None, :] + offsets[:, None] * normal[1]
    sampled = cv2.remap(
        top_hat.astype(np.float32),
        sample_x.astype(np.float32),
        sample_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    profile = np.quantile(sampled, 0.65, axis=1)
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 5), 0).ravel()
    return offsets, profile


def _visible_edge_offset(
    offsets: np.ndarray,
    profile: np.ndarray,
    *,
    maximum_band_width: float = 12.0,
    mode: str = "outer_edge",
) -> tuple[float, dict[str, float]] | None:
    """Locate a painted line in a profile sampled normal to it.

    ``mode="outer_edge"`` returns the far end of the painted band along the normal, which is
    the visible edge the refinement is asked to move the line onto. ``mode="centre"`` returns
    the response-weighted centroid of the band, which is the sub-pixel centre of the paint.
    """
    peak_index = int(np.argmax(profile))
    baseline = float(np.quantile(profile, 0.20))
    peak = float(profile[peak_index])
    prominence = peak - baseline
    if prominence < 5.0:
        return None
    threshold = baseline + max(3.0, 0.24 * prominence)
    left = peak_index
    right = peak_index
    while left > 0 and profile[left - 1] >= threshold:
        left -= 1
    while right + 1 < len(profile) and profile[right + 1] >= threshold:
        right += 1
    band_width = float(offsets[right] - offsets[left])
    if band_width > maximum_band_width or right == len(profile) - 1:
        return None
    if mode == "centre":
        if left == 0:
            return None
        weights = profile[left : right + 1] - threshold
        edge_offset = float(np.sum(weights * offsets[left : right + 1]) / np.sum(weights))
    else:
        edge_offset = float(offsets[right])
    return edge_offset, {
        "edge_offset_px": edge_offset,
        "peak_offset_px": float(offsets[peak_index]),
        "peak_response": peak,
        "profile_baseline": baseline,
        "prominence": prominence,
        "band_width_px": band_width,
    }


def _fit_painted_line_in_corridor(
    top_hat: np.ndarray,
    endpoints: np.ndarray,
    orientation: str,
    *,
    corridor_pixels: float,
    edge_convention: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]] | None:
    scale = _pixel_scale(top_hat.shape[1])
    initial = _segment_line(endpoints.ravel())
    initial_normal = _oriented_line_normal(initial, orientation, edge_convention)
    direction = endpoints[1] - endpoints[0]
    length = float(np.linalg.norm(direction))
    if length < 40.0 * scale:
        return None
    direction /= length
    base_points = np.linspace(endpoints[0], endpoints[1], 96)[6:-6]
    offsets = np.arange(-corridor_pixels, corridor_pixels + 0.5, 0.5)
    sample_x = base_points[:, 0][None, :] + offsets[:, None] * initial_normal[0]
    sample_y = base_points[:, 1][None, :] + offsets[:, None] * initial_normal[1]
    sampled = cv2.remap(
        top_hat.astype(np.float32),
        sample_x.astype(np.float32),
        sample_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    peak_indices = np.argmax(sampled, axis=0)
    peak_offsets = offsets[peak_indices]
    peak_responses = sampled[peak_indices, np.arange(sampled.shape[1])]
    reliable = peak_responses >= 18.0
    if reliable.sum() < 24:
        return None
    median_offset = float(np.median(peak_offsets[reliable]))
    absolute_deviation = np.abs(peak_offsets - median_offset)
    mad = float(np.median(absolute_deviation[reliable]))
    reliable &= absolute_deviation <= max(2.0 * scale, 3.0 * mad)
    points = (
        base_points[reliable] + peak_offsets[reliable, None] * initial_normal[None, :]
    ).astype(np.float32)
    if len(points) < 24:
        return None
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    fitted = np.asarray([vy, -vx, vx * y0 - vy * x0], dtype=float)
    fitted /= np.hypot(fitted[0], fitted[1])
    fitted_normal = _oriented_line_normal(fitted, orientation, edge_convention)
    sign = 1.0 if np.dot(fitted[:2], fitted_normal) >= 0 else -1.0
    fitted *= sign
    angle_delta = abs(_angle(fitted) - _angle(initial))
    endpoint_distances = endpoints @ fitted[:2] + fitted[2]
    center_shift = float(np.median(endpoint_distances))
    if angle_delta > 0.6 or abs(center_shift) > min(5.0 * scale, corridor_pixels):
        return None
    fitted_endpoints = endpoints - endpoint_distances[:, None] * fitted[:2]
    return (
        fitted,
        fitted_endpoints,
        {
            "local_fit_points": len(points),
            "local_fit_angle_delta_degrees": float(angle_delta),
            "local_fit_center_shift_px": center_shift,
            "local_fit_peak_offset_median_px": median_offset,
            "local_fit_peak_offset_mad_px": mad,
        },
    )


def refine_visible_painted_edges(
    image: np.ndarray,
    homography: np.ndarray,
    *,
    maximum_offset: float = 12.0,
    offset_step: float = 0.5,
    minimum_refined_lines: int = 5,
    return_rejected_candidate: bool = False,
    fit_line_orientation: bool = False,
    edge_convention: str | None = None,
    witness_convention: str | None = None,
    witness_mask: str = "standard",
) -> tuple[np.ndarray, dict[str, object]]:
    """Refine a solved topology to the visible outer edge of each painted line.

    The topology solver establishes line identity. This step searches only normal to each
    identified line, retains the original line when local image evidence is weak, and fails
    closed when the resulting projective court loses independent topology or surface support.
    The visible-edge direction follows the ``itf_outside_edge_v1`` labeling convention.

    ``maximum_offset`` is given in canonical 960x540 pixels and is scaled to the image, so
    the corridor covers the same distance on court whatever resolution the solve runs at.
    ``offset_step`` is in image pixels: a native solve samples the profile at the same
    physical fineness as the sensor, which is where its extra sub-pixel precision comes from.
    ``witness_mask`` preserves the proposal arm for both original and refined paint scores;
    edge/paint conventions remain explicitly controlled by ``witness_convention``.
    """
    if witness_mask not in {"standard", "relaxed_chroma"}:
        raise ValueError(f"unknown refinement witness mask {witness_mask!r}")
    edge_convention = resolve_edge_convention(edge_convention)
    witness_convention = resolve_refinement_witness(witness_convention)
    height, width = image.shape[:2]
    scale = _pixel_scale(width)
    maximum_offset = maximum_offset * scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel_size = max(9, int(round(15 * width / 960)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    top_hat = cv2.morphologyEx(
        gray,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)),
    )
    refined_lines: dict[str, np.ndarray] = {}
    line_evidence: dict[str, dict[str, float | bool]] = {}
    refined_count = 0
    for name, start, end, orientation in VISIBLE_EDGE_WORLD_LINES:
        endpoints = _project_court_points(homography, np.asarray([start, end]))
        line = _segment_line(endpoints.ravel())
        local_evidence = {}
        if fit_line_orientation:
            local_fit = _fit_painted_line_in_corridor(
                top_hat,
                endpoints,
                orientation,
                corridor_pixels=maximum_offset,
                edge_convention=edge_convention,
            )
            if local_fit is not None:
                line, endpoints, local_evidence = local_fit
        points = np.linspace(endpoints[0], endpoints[1], 120)[10:-10]
        normal = _oriented_line_normal(line, orientation, edge_convention)
        offsets, profile = _painted_edge_profile(
            top_hat,
            points,
            normal,
            maximum_offset=maximum_offset,
            offset_step=offset_step,
        )
        result = _visible_edge_offset(
            offsets,
            profile,
            maximum_band_width=12.0 * scale,
            mode="centre" if edge_convention == "paint_centre" else "outer_edge",
        )
        if result is None:
            refined_lines[name] = line
            line_evidence[name] = {"refined": False, **local_evidence}
            continue
        edge_offset, evidence = result
        shifted = line.copy()
        shifted[:2] = normal
        shifted[2] = line[2] * (1.0 if np.dot(line[:2], normal) >= 0 else -1.0) - edge_offset
        refined_lines[name] = shifted
        line_evidence[name] = {"refined": True, **local_evidence, **evidence}
        refined_count += 1
    if refined_count < minimum_refined_lines:
        return homography, {
            "accepted": False,
            "reason": "insufficient_refined_lines",
            "refined_lines": refined_count,
            "lines": line_evidence,
        }
    source = []
    target = []
    horizontal_names = (
        ("near_baseline", 0.0),
        ("near_service", 5.485),
        ("far_service", 18.285),
        ("far_baseline", court.COURT_L),
    )
    for horizontal_name, court_y in horizontal_names:
        for vertical_name, court_x in (("left_doubles", 0.0), ("right_doubles", court.COURT_W)):
            point = _intersection(refined_lines[horizontal_name], refined_lines[vertical_name])
            if point is None:
                return homography, {
                    "accepted": False,
                    "reason": "parallel_refined_lines",
                    "refined_lines": refined_count,
                    "lines": line_evidence,
                }
            source.append(point)
            target.append((court_x, court_y))
    world_to_image, _ = cv2.findHomography(
        np.asarray(target, np.float32),
        np.asarray(source, np.float32),
        0,
    )
    if world_to_image is None:
        fitted = None
    else:
        fitted = np.linalg.inv(world_to_image)
    if fitted is None or not _standard_broadcast_view(fitted, width, height):
        return homography, {
            "accepted": False,
            "reason": "invalid_refined_geometry",
            "refined_lines": refined_count,
            "lines": line_evidence,
        }
    # Keep the accepted proposal's paint observation operator through refinement.
    # A relaxed-chroma proposal must not be judged by absent standard-mask paint.
    mask = (
        relaxed_chroma_line_mask(image)
        if witness_mask == "relaxed_chroma"
        else court.line_mask(image, top_fraction=0.12)
    )
    distance = _distance_samples(mask)
    # The seed is a line-centre fit and the candidate is an outside-edge fit, so each is
    # scored under its own convention. Scoring both as nominal costs the candidate half a
    # line width of distance-to-paint for doing exactly what it is asked to do.
    original_topology = topology_score(homography, mask, distance=distance)
    refined_topology = topology_score(
        fitted,
        mask,
        distance=distance,
        convention=witness_convention,
        edge_convention=edge_convention,
    )
    nominal_refined_topology = topology_score(fitted, mask, distance=distance)
    original_surface = court_surface_consistency(image, homography)
    refined_surface = court_surface_consistency(image, fitted)
    fitted_pixels = cv2.perspectiveTransform(
        np.asarray(target, np.float32).reshape(1, -1, 2),
        world_to_image,
    )[0]
    fit_residuals = np.linalg.norm(fitted_pixels - np.asarray(source), axis=1)
    accepted = (
        refined_topology >= original_topology - 0.01 and refined_surface >= original_surface - 0.04
    )
    evidence = {
        "accepted": accepted,
        "reason": "accepted" if accepted else "independent_witness_regression",
        "refined_lines": refined_count,
        "witness_mask": witness_mask,
        "original_topology_convention": "nominal",
        "refined_topology_convention": witness_convention,
        "original_topology_score": original_topology,
        "refined_topology_score": refined_topology,
        "refined_topology_score_nominal_convention": nominal_refined_topology,
        "edge_convention": edge_convention,
        "witness_convention": witness_convention,
        "original_surface_score": original_surface,
        "refined_surface_score": refined_surface,
        "projective_fit_median_px": float(np.median(fit_residuals)),
        "projective_fit_maximum_px": float(np.max(fit_residuals)),
        "lines": line_evidence,
    }
    return (
        fitted if accepted or return_rejected_candidate else homography,
        evidence,
    )


def refine_from_topology(
    homography: np.ndarray,
    horizontal: list[LineHypothesis],
    vertical: list[LineHypothesis],
    width: int,
    height: int,
) -> np.ndarray:
    current = homography
    for _ in range(2):
        projected_h = [
            _projected_line(current, (0.0, y), (court.COURT_W, y)) for y in HORIZONTAL_WORLD
        ]
        projected_v = [
            _projected_line(current, (x, 0.0), (x, court.COURT_L)) for x in VERTICAL_WORLD
        ]
        h_cost = np.asarray(
            [
                [_line_cost(row, model, "horizontal", width, height) for model in projected_h]
                for row in horizontal
            ]
        )
        v_cost = np.asarray(
            [
                [_line_cost(row, model, "vertical", width, height) for model in projected_v]
                for row in vertical
            ]
        )
        h_rows, h_cols = linear_sum_assignment(h_cost)
        v_rows, v_cols = linear_sum_assignment(v_cost)
        h_matches = [
            (horizontal[row], HORIZONTAL_WORLD[col])
            for row, col in zip(h_rows, h_cols)
            if h_cost[row, col] <= 2.5
        ]
        v_matches = [
            (vertical[row], VERTICAL_WORLD[col])
            for row, col in zip(v_rows, v_cols)
            if v_cost[row, col] <= 2.5
        ]
        source = []
        target = []
        for h_line, court_y in h_matches:
            for v_line, court_x in v_matches:
                point = _intersection(h_line.line, v_line.line)
                if point is None:
                    continue
                source.append(point)
                target.append((court_x, court_y))
        if len(source) < 6 or len(h_matches) < 2 or len(v_matches) < 2:
            break
        fitted, _ = cv2.findHomography(
            np.asarray(source, np.float32), np.asarray(target, np.float32), 0
        )
        if fitted is None or not assess_court_homography(fitted, width, height)["valid"]:
            break
        current = fitted
    return current


def court_topology_hypotheses(
    image: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    spatially_balanced: bool = False,
    limit: int = 12,
    proposal_pool: int = 48,
) -> tuple[CourtTopologyHypothesis, ...]:
    mask, horizontal, vertical = detect_line_families(image, mask=mask)
    height, width = mask.shape
    distance = _distance_samples(mask)
    if spatially_balanced:
        horizontal = _spatially_balanced_hypotheses(horizontal, height)
        vertical = _spatially_balanced_hypotheses(vertical, width)
    else:
        horizontal = horizontal[:10]
        vertical = vertical[:10]
    top = [row for row in horizontal if row.reference <= 0.58 * height]
    bottom = [row for row in horizontal if row.reference >= 0.42 * height]
    left = [row for row in vertical if row.reference <= 0.58 * width]
    right = [row for row in vertical if row.reference >= 0.42 * width]
    if not top or not bottom or not left or not right:
        raise ValueError("insufficient merged court-line families")
    horizontal_pairs = (
        (far, near)
        for far in top
        for near in bottom
        if near.reference - far.reference >= 0.10 * height
    )
    vertical_pairs = (
        (left_line, right_line)
        for left_line in left
        for right_line in right
        if right_line.reference - left_line.reference >= 0.16 * width
    )
    world_h_pairs = [(far, near) for near in HORIZONTAL_WORLD[:2] for far in HORIZONTAL_WORLD[2:]]
    world_v_pairs = [
        (left_x, right_x) for left_x in VERTICAL_WORLD[:2] for right_x in VERTICAL_WORLD[2:]
    ]
    ranked: list[tuple[float, int, np.ndarray]] = []
    candidate_id = 0
    vertical_pairs = list(vertical_pairs)
    for far_line, near_line in horizontal_pairs:
        for left_line, right_line in vertical_pairs:
            image_points = []
            valid = True
            for horizontal_line in (near_line, far_line):
                for vertical_line in (left_line, right_line):
                    point = _intersection(horizontal_line.line, vertical_line.line)
                    if point is None:
                        valid = False
                        break
                    image_points.append(point)
                if not valid:
                    break
            if not valid:
                continue
            for far_y, near_y in world_h_pairs:
                for left_x, right_x in world_v_pairs:
                    world_points = [
                        (left_x, near_y),
                        (right_x, near_y),
                        (left_x, far_y),
                        (right_x, far_y),
                    ]
                    candidate = cv2.getPerspectiveTransform(
                        np.asarray(image_points, np.float32),
                        np.asarray(world_points, np.float32),
                    )
                    if not _standard_broadcast_view(candidate, width, height):
                        continue
                    score = _vertex_score(candidate, distance)
                    candidate_id += 1
                    heapq.heappush(ranked, (score, candidate_id, candidate))
                    if len(ranked) > proposal_pool:
                        heapq.heappop(ranked)
    if not ranked:
        return ()
    hypotheses = []
    for _, _, candidate in sorted(ranked, reverse=True):
        refined = refine_from_topology(candidate, horizontal, vertical, width, height)
        for option in (refined, candidate):
            if not _standard_broadcast_view(option, width, height):
                continue
            score = topology_score(option, mask, distance=distance)
            hypotheses.append(CourtTopologyHypothesis(option, score))
    hypotheses.sort(key=lambda row: row.score, reverse=True)
    distinct = []
    for hypothesis in hypotheses:
        projected = _project_court_points(hypothesis.homography, np.asarray(LANDMARKS))
        if any(
            np.median(
                np.linalg.norm(
                    projected - _project_court_points(existing.homography, np.asarray(LANDMARKS)),
                    axis=1,
                )
            )
            <= 2.0 * _pixel_scale(width)
            for existing in distinct
        ):
            continue
        distinct.append(hypothesis)
        if len(distinct) >= limit:
            break
    return tuple(distinct)


def find_court_h_topology(
    image: np.ndarray,
    *,
    minimum_score: float = 0.80,
    mask: np.ndarray | None = None,
    spatially_balanced: bool = False,
) -> np.ndarray:
    hypotheses = court_topology_hypotheses(
        image,
        mask=mask,
        spatially_balanced=spatially_balanced,
        limit=1,
    )
    best_score = hypotheses[0].score if hypotheses else -1.0
    if not hypotheses or best_score < minimum_score:
        raise ValueError(f"no supported court topology (best={best_score:.3f})")
    return hypotheses[0].homography


def find_court_h_topology_multiframe(
    target_image: np.ndarray,
    context_images: list[np.ndarray],
    *,
    minimum_target_score: float = 0.80,
) -> np.ndarray:
    """Recover a static-camera calibration from nearby frames without weakening its gate.

    Every source frame must independently pass the strict single-frame solver. Its homography
    is reusable only when the target frame also supports the same full topology, which rejects
    camera moves and cuts rather than silently transferring stale geometry.
    """
    target_mask = court.line_mask(target_image, top_fraction=0.12)
    candidates = []
    for source_index, source_image in enumerate([target_image, *context_images]):
        try:
            homography = find_court_h_topology(source_image)
        except (ValueError, np.linalg.LinAlgError, cv2.error):
            continue
        source_mask = court.line_mask(source_image, top_fraction=0.12)
        source_score = topology_score(homography, source_mask)
        target_score = topology_score(homography, target_mask)
        if target_score >= minimum_target_score and _standard_broadcast_view(
            homography,
            target_image.shape[1],
            target_image.shape[0],
        ):
            candidates.append((min(source_score, target_score), -source_index, homography))
    if not candidates:
        raise ValueError("no camera-stable multiframe court topology")
    return max(candidates, key=lambda row: (row[0], row[1]))[2]
