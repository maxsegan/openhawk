"""Put a supported camera whose court lines miss the paint back on the paint.

The automatic registration (``court.find_court_h``) scores a line-to-model assignment by
aggregate support at 960x540. A hypothesis one band off can win it: on Monte Carlo 2025
the camera draws its far service line on the painted far baseline, and on Rome 2026 some
shots draw the far baseline on the painted far service line. The far player then stands
metres from where the fit puts the hit, and most such flights fail their gates.

This module tests every projected court line against the paint at native resolution. A
camera whose lines are all on the paint is left alone. Otherwise it re-enumerates the
line-to-model assignments on that frame, scores each one per line (every visible line has
to be on the paint, not an average), refits the ground homography to the painted line
points, and keeps the new projection only when all visible lines are on the paint and the
worst line improved clearly. It never reads a clicked court or a label. In ``on`` mode a
frame it cannot fix keeps its camera; this is a repair, not a gate.

``abstain`` mode repairs the same way and then verifies every contiguous run of frames that
shares a projection on up to five spread frames. A frame is on the paint when all four
court-width lines (baselines and service lines, which fix depth) reach ``TRIGGER``, or when
its worst deciding line reaches ``VERIFY_DECIDING`` and the median of all its visible lines
reaches ``TRIGGER``; a run whose scored frames are mostly off the paint is marked
unsupported instead of fitting flights at a wrong far-court depth (Rome 2026 pt0026: one
static-fallback camera over a moving broadcast camera; a Hawk-Eye graphic given an
interpolated court camera). Neither the worst sideline nor one marginal deciding line
decides alone: good broadcast cameras show occluded or unpainted sidelines, and a baseline
partly hidden by a player or the picture edge scores about 0.4 on an otherwise exact camera.
"""

from __future__ import annotations

import itertools
from typing import Mapping

import cv2
import numpy as np

from cv.pipeline import court
from cv.pipeline.camera_cal import (
    fixed_f_projection_from_ground,
    intrinsic_projection_from_ground,
    net_reprojection_check,
)
from cv.pipeline.court_near_baseline_refinement import _load_image

MODE_OFF = "off"
MODE_ON = "on"
MODE_ABSTAIN = "abstain"
MODES = (MODE_OFF, MODE_ON, MODE_ABSTAIN)

_SW = (court.COURT_W - 8.23) / 2.0
_NEAR_SERVICE = court.NET_Y - court.SERVICE_FROM_NET
_FAR_SERVICE = court.NET_Y + court.SERVICE_FROM_NET
#: Model lines tested against the paint. The near service line is reported but does not
#: decide: clay slides wear it off on most broadcasts.
LINES = {
    "near_baseline": ((0.0, 0.0), (court.COURT_W, 0.0)),
    "far_baseline": ((0.0, court.COURT_L), (court.COURT_W, court.COURT_L)),
    "near_service": ((_SW, _NEAR_SERVICE), (court.COURT_W - _SW, _NEAR_SERVICE)),
    "far_service": ((_SW, _FAR_SERVICE), (court.COURT_W - _SW, _FAR_SERVICE)),
    "left_doubles": ((0.0, 0.0), (0.0, court.COURT_L)),
    "right_doubles": ((court.COURT_W, 0.0), (court.COURT_W, court.COURT_L)),
    "left_singles": ((_SW, 0.0), (_SW, court.COURT_L)),
    "right_singles": ((court.COURT_W - _SW, 0.0), (court.COURT_W - _SW, court.COURT_L)),
    "centre": ((court.COURT_W / 2.0, _NEAR_SERVICE), (court.COURT_W / 2.0, _FAR_SERVICE)),
}
DECIDING = ("near_baseline", "far_baseline", "far_service")
_LATERAL = ("left_doubles", "right_doubles", "left_singles", "right_singles", "centre")
_H_CANDS = (0.0, _NEAR_SERVICE, _FAR_SERVICE, court.COURT_L)
_V_CANDS = ((0.0, court.COURT_W), (_SW, court.COURT_W - _SW))
_SAMPLES = 48
_MIN_VISIBLE = 12
_CONTRAST = 18.0
#: A camera is left alone unless a visible deciding line is below this hit fraction.
TRIGGER = 0.40
#: A refit must put every required line (the deciding lines the camera showed, all four
#: sidelines and the centre line) at or above this hit fraction.
ON_PAINT = 0.55
#: A refit is kept only when its worst deciding line is on the paint, every visible
#: sideline too, and the worst deciding line gained at least this much.
MIN_GAIN = 0.30
#: Frames in one projection run: a larger frame gap starts a new run in ``abstain`` mode.
RUN_GAP = 3
#: Frames verified per run in ``abstain`` mode.
VERIFY_FRAMES = 5
#: ``abstain`` verification: the worst deciding line may sit this low when the median
#: visible line is on the paint (a partly hidden baseline scores 0.38-0.44 on exact cameras).
VERIFY_DECIDING = 0.30
_WIDTH_LINES = ("near_baseline", "far_baseline", "near_service", "far_service")
#: Refits tried per camera document, largest groups first (a refit costs ~5 s). Labelled
#: attempts need at most 14; a camera redrawn every frame (Rome pt0067: 195 groups, 331
#: refits, ~30 min) otherwise exhausts the attempt timeout before any flight is fitted.
MAX_REFITS = 80

_SAMPLE_T = np.linspace(0.04, 0.96, _SAMPLES)
_MODEL_POINTS = {
    name: np.asarray(a, float)[None, :]
    + _SAMPLE_T[:, None] * (np.asarray(b) - np.asarray(a))[None, :]
    for name, (a, b) in LINES.items()
}


def validate_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError("explicit court paint refinement mode required")
    return mode


def paint_mask(image: np.ndarray) -> np.ndarray:
    """Thin bright low-saturation paint (top-hat), dilated by two pixels; line candidates only."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = max(15, int(round(image.shape[1] / 64.0))) | 1
    hat = cv2.morphologyEx(
        gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    )
    mask = ((hat >= 22) & (gray >= 110) & (hsv[..., 1] < 110)).astype(np.uint8)
    return cv2.dilate(mask, np.ones((5, 5), np.uint8))


def _to_image(court_to_image: np.ndarray, points: np.ndarray) -> np.ndarray:
    h = np.c_[points, np.ones(len(points))] @ court_to_image.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return h[:, :2] / h[:, 2:3]


def ground_homography(projection: np.ndarray) -> np.ndarray:
    """Court-plane (x, y, 1) to image homography of a 3x4 projection."""
    return np.asarray(projection, float)[:, [0, 1, 3]]


def line_scores(gray: np.ndarray, court_to_image: np.ndarray) -> dict[str, float | None]:
    """Hit fraction per model line; ``None`` when too little of the line is in the picture.

    A sample is on the paint when the brightest pixel within 2 px across the projected line
    is at least ``_CONTRAST`` grey levels above both sides 9 px away: a thin bright stripe,
    not a bright region. Crowd, boards and sky rarely pass it along a whole line.
    """
    height, width = gray.shape
    out: dict[str, float | None] = {}
    for name, points in _MODEL_POINTS.items():
        pixels = _to_image(court_to_image, points)
        if not np.isfinite(pixels).all():
            out[name] = None
            continue
        direction = pixels[-1] - pixels[0]
        length = float(np.linalg.norm(direction))
        if length < 20:
            out[name] = None
            continue
        normal = np.array([-direction[1], direction[0]]) / length
        offsets = np.array([-9.0, -2.0, -1.0, 0.0, 1.0, 2.0, 9.0])
        probes = pixels[:, None, :] + offsets[None, :, None] * normal[None, None, :]
        inside = (
            (probes[..., 0] >= 1).all(axis=1)
            & (probes[..., 0] < width - 1).all(axis=1)
            & (probes[..., 1] >= 1).all(axis=1)
            & (probes[..., 1] < height - 1).all(axis=1)
        )
        if int(inside.sum()) < _MIN_VISIBLE:
            out[name] = None
            continue
        xy = np.rint(probes[inside]).astype(int)
        values = gray[xy[..., 1], xy[..., 0]].astype(float)
        stripe = values[:, 1:6].max(axis=1)
        sides = np.maximum(values[:, 0], values[:, 6])
        out[name] = float(((stripe - sides) >= _CONTRAST).mean())
    return out


def worst_deciding(scores: Mapping[str, float | None]) -> float | None:
    values = [scores[name] for name in DECIDING if scores.get(name) is not None]
    return min(values) if values else None


def worst_required(scores: Mapping[str, float | None], required: tuple[str, ...]) -> float:
    """Worst hit fraction over ``required``; a line pushed out of the picture counts as 0."""
    return min((scores.get(name) or 0.0) for name in required) if required else 0.0


def _court_region(court_to_image: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """The current camera's court widened by 3 m and lengthened by 4 m at each end.

    A camera one band off still covers the painted court with this margin, and the crowd,
    boards and scoreboard above it no longer compete for line candidates.
    """
    corners = np.array(
        [
            [-3.0, -4.0],
            [court.COURT_W + 3.0, -4.0],
            [court.COURT_W + 3.0, court.COURT_L + 4.0],
            [-3.0, court.COURT_L + 4.0],
        ]
    )
    polygon = _to_image(court_to_image, corners)
    region = np.zeros(shape[:2], np.uint8)
    if np.isfinite(polygon).all():
        cv2.fillPoly(region, [np.rint(polygon).astype(np.int32)], 1)
    return region


def _segments(mask: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Long paint lines in normal form, split into horizontal and steep families."""
    height, width = mask.shape
    raw = cv2.HoughLinesP(
        mask * 255,
        1,
        np.pi / 360.0,
        threshold=max(60, width // 24),
        minLineLength=width // 16,
        maxLineGap=max(12, width // 90),
    )
    if raw is None:
        return [], []
    clusters: list[list] = []
    for x1, y1, x2, y2 in raw.reshape(-1, 4).astype(float):
        direction = np.array([x2 - x1, y2 - y1])
        length = float(np.linalg.norm(direction))
        if length < 1:
            continue
        normal = np.array([-direction[1], direction[0]]) / length
        if normal[1] < 0 or (abs(normal[1]) < 1e-9 and normal[0] < 0):
            normal = -normal
        rho = float(normal @ np.array([x1, y1]))
        for cluster in clusters:
            if (
                abs(float(cluster[0] @ normal)) > np.cos(np.radians(2.0))
                and abs(cluster[1] - rho) < 6.0
            ):
                cluster[2] += length
                break
        else:
            clusters.append([normal, rho, length])
    clusters.sort(key=lambda item: -item[2])
    horizontal, steep = [], []
    for normal, rho, _length in clusters:
        line = np.array([normal[0], normal[1], -rho])
        angle = abs(np.degrees(np.arctan2(normal[0], normal[1])))
        angle = min(angle, 180.0 - angle)
        if angle < 18.0 and len(horizontal) < 12:
            horizontal.append(line)
        elif angle > 30.0 and len(steep) < 10:
            steep.append(line)
    return horizontal, steep


def _meet(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    point = np.cross(first, second)
    if abs(float(point[2])) < 1e-9:
        return None
    return point[:2] / point[2]


def _row_at_centre(line: np.ndarray, width: int) -> float:
    return float(-(line[0] * width / 2.0 + line[2]) / line[1])


def _column_at(line: np.ndarray, row: float) -> float:
    return float(-(line[1] * row + line[2]) / line[0])


def _polish(mask: np.ndarray, court_to_image: np.ndarray) -> np.ndarray:
    """Least-squares homography through paint pixels within 5 px of each projected line."""
    court_points, image_points = [], []
    raw = cv2.findNonZero(cv2.erode(mask, np.ones((3, 3), np.uint8)))
    if raw is None:
        return court_to_image
    paint = raw.reshape(-1, 2).astype(float)
    for a, b in LINES.values():
        a, b = np.asarray(a, float), np.asarray(b, float)
        pa, pb = _to_image(court_to_image, np.vstack([a, b]))
        direction = pb - pa
        length = float(np.linalg.norm(direction))
        if not np.isfinite(length) or length < 20:
            continue
        normal = np.array([-direction[1], direction[0]]) / length
        distance = np.abs((paint - pa) @ normal)
        along = ((paint - pa) @ direction) / length**2
        near = (distance <= 5.0) & (along > 0.03) & (along < 0.97)
        if int(near.sum()) < 20:
            continue
        chosen = np.flatnonzero(near)[:: max(1, int(near.sum()) // 60)]
        for index in chosen:
            t = float(np.clip(along[index], 0.0, 1.0))
            court_points.append(a + t * (b - a))
            image_points.append(paint[index])
    if len(court_points) < 24:
        return court_to_image
    solved, _ = cv2.findHomography(
        np.asarray(court_points, np.float32), np.asarray(image_points, np.float32), cv2.RANSAC, 3.0
    )
    return court_to_image if solved is None else solved


def refine_projection(image: np.ndarray, projection: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """Return a projection on the paint, or ``None`` with the reason the camera stays."""
    if image.ndim != 3 or projection.shape != (3, 4) or not np.isfinite(projection).all():
        return None, {"status": "kept", "reason": "bad_input"}
    height, width = image.shape[:2]
    mask = paint_mask(image) * _court_region(ground_homography(projection), image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    before = line_scores(gray, ground_homography(projection))
    worst_before = worst_deciding(before)
    receipt: dict = {"before": {k: None if v is None else round(v, 3) for k, v in before.items()}}
    if worst_before is None:
        return None, {**receipt, "status": "kept", "reason": "no_visible_deciding_line"}
    if worst_before >= TRIGGER:
        return None, {**receipt, "status": "on_paint", "reason": "lines_on_paint"}
    # Every sideline and the centre line are required too: a hypothesis that lays the
    # model's doubles line on a painted singles line is on the paint there but skewed.
    required = (
        *(name for name in DECIDING if before.get(name) is not None),
        *_LATERAL,
    )
    horizontal, steep = _segments(mask)
    if len(horizontal) < 2 or len(steep) < 2:
        return None, {**receipt, "status": "kept", "reason": "too_few_paint_lines"}
    horizontal.sort(key=lambda line: _row_at_centre(line, width))
    best = None
    for upper, lower in itertools.combinations(horizontal, 2):
        row_upper, row_lower = _row_at_centre(upper, width), _row_at_centre(lower, width)
        if row_lower - row_upper < 0.08 * height:
            continue
        for low_index, high_index in itertools.combinations(range(4), 2):
            # The upper image line is farther from the camera: the larger court y.
            ys = {id(upper): _H_CANDS[high_index], id(lower): _H_CANDS[low_index]}
            for left, right in itertools.permutations(steep, 2):
                if _column_at(left, row_lower) >= _column_at(right, row_lower) - 0.1 * width:
                    continue
                for xl, xr in _V_CANDS:
                    image_pts, court_pts = [], []
                    for hline in (upper, lower):
                        for vline, x in ((left, xl), (right, xr)):
                            point = _meet(hline, vline)
                            if point is None or not (
                                -width <= point[0] <= 2 * width
                                and -height <= point[1] <= 2 * height
                            ):
                                break
                            image_pts.append(point)
                            court_pts.append((x, ys[id(hline)]))
                    if len(image_pts) != 4:
                        continue
                    candidate = cv2.getPerspectiveTransform(
                        np.asarray(court_pts, np.float32), np.asarray(image_pts, np.float32)
                    )
                    if candidate is None or not np.isfinite(candidate).all():
                        continue
                    scores = line_scores(gray, candidate)
                    worst = worst_required(scores, required)
                    visible = [v for v in scores.values() if v is not None]
                    key = (round(worst, 2), float(np.mean(visible)), len(visible))
                    if best is None or key > best[0]:
                        best = (key, candidate)
    if best is None:
        return None, {**receipt, "status": "kept", "reason": "no_assignment"}
    court_to_image = _polish(mask, best[1])
    after = line_scores(gray, court_to_image)
    if worst_required(after, required) < worst_required(line_scores(gray, best[1]), required):
        court_to_image, after = best[1], line_scores(gray, best[1])
    worst_after = worst_required(after, required)
    receipt["after"] = {k: None if v is None else round(v, 3) for k, v in after.items()}
    receipt["required"] = list(required)
    if worst_after < ON_PAINT or worst_after - worst_before < MIN_GAIN:
        return None, {**receipt, "status": "kept", "reason": "refit_not_on_paint"}
    homography = court_to_image / court_to_image[2, 2]
    solved = intrinsic_projection_from_ground(homography, w=width, h=height)
    if solved is None:
        # Self-calibration can fail on a near-frontal view. Borrow the focal length the
        # supplied camera was solved with, and keep the result only if the net tape still
        # projects to a plausible height.
        previous = intrinsic_projection_from_ground(
            ground_homography(projection) / projection[2, 3], w=width, h=height
        )
        if previous is None:
            return None, {**receipt, "status": "kept", "reason": "intrinsic_solve_failed"}
        candidate = fixed_f_projection_from_ground(homography, previous[1], w=width, h=height)
        check = net_reprojection_check(candidate, homography)
        if check["ground_err_px"] > 0.1 or not all(3.0 <= px <= 200.0 for px in check["net_px"]):
            return None, {**receipt, "status": "kept", "reason": "intrinsic_solve_failed"}
        solved = (candidate, previous[1])
    refined, focal = solved
    return refined, {
        **receipt,
        "status": "refit",
        "reason": "lines_back_on_paint",
        "focal_px": round(float(focal), 1),
    }


def _group_key(projection: np.ndarray) -> tuple[int, ...]:
    corners = _to_image(
        ground_homography(projection),
        np.array(
            [[0, 0], [court.COURT_W, 0], [0, court.COURT_L], [court.COURT_W, court.COURT_L]], float
        ),
    )
    return tuple(int(round(value / 6.0)) for value in corners.ravel())


SUPPLIED_P_FIELD = "court_paint_supplied_P"
#: An abstained row keeps the values it replaced here (``P``, ``supported``, ``status``, ``reason``).
SUPPLIED_ROW_FIELD = "court_paint_supplied_row"
_ABSTAIN_KEYS = ("P", "supported", "status", "reason")


def frame_on_paint(gray: np.ndarray, projection: np.ndarray) -> bool | None:
    """Whether one frame's court is on the paint; ``None`` if no deciding line is scorable."""
    return scores_on_paint(line_scores(gray, ground_homography(projection)))


def scores_on_paint(scores: Mapping[str, float | None]) -> bool | None:
    """All four court-width lines on the paint, or the worst deciding line at least
    ``VERIFY_DECIDING`` with the median visible line on the paint."""
    deciding = worst_deciding(scores)
    if deciding is None:
        return None
    width = [scores.get(name) for name in _WIDTH_LINES]
    if all(value is not None and value >= TRIGGER for value in width):
        return True
    seen = [value for value in scores.values() if value is not None]
    return deciding >= VERIFY_DECIDING and float(np.median(seen)) >= TRIGGER


def _runs(rows: list[dict], indexes: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for index in sorted(indexes, key=lambda item: int(rows[item]["frame"])):
        if runs and int(rows[index]["frame"]) - int(rows[runs[-1][-1]]["frame"]) <= RUN_GAP:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def strip_refinement(document: dict) -> dict | None:
    """The document this refinement was applied to, or ``None`` if it was not only refits.

    Every refit row carries the projection it replaced, so the supplied camera is recovered
    exactly and the component plan can still be checked against the file it was hashed from.
    """
    if document.get("court_paint_refinement") not in (MODE_ON, MODE_ABSTAIN):
        return document
    base = {k: v for k, v in document.items() if k != "court_paint_refinement"}
    rows = []
    for row in document.get("cameras") or []:
        if row.get("court_paint_geometry") is None and SUPPLIED_P_FIELD not in row:
            rows.append(row)
            continue
        if row.get("court_paint_geometry") == "abstained":
            supplied = row.get(SUPPLIED_ROW_FIELD)
            if not isinstance(supplied, dict) or row.get("supported") is not False:
                return None
            restored = {
                k: v
                for k, v in row.items()
                if k not in {"court_paint_geometry", SUPPLIED_ROW_FIELD, *_ABSTAIN_KEYS}
            }
            restored.update(supplied)
            rows.append(restored)
            continue
        if row.get("court_paint_geometry") != "refit" or not isinstance(
            row.get(SUPPLIED_P_FIELD), list
        ):
            return None
        if not row.get("supported") or not isinstance(row.get("P"), list):
            return None
        restored = {
            k: v for k, v in row.items() if k not in {"court_paint_geometry", SUPPLIED_P_FIELD}
        }
        restored["P"] = row[SUPPLIED_P_FIELD]
        rows.append(restored)
    base["cameras"] = rows
    return base


def refine_camera_document(
    document: dict,
    images: Mapping[int, np.ndarray] | None = None,
    *,
    mode: str,
    frame_paths: Mapping[int, str] | None = None,
) -> tuple[dict, dict]:
    """Refit supported rows whose court is off the paint. ``off`` returns the document unchanged.

    Rows are grouped by their projected court corners (6 px bins). Each group is tested on
    up to three of its frames; a refit found on one frame is kept only if it is also on the
    paint on a second frame of the group, and then replaces every row of the group.
    ``abstain`` then marks unsupported each run of a group that stays off the paint.
    """
    validate_mode(mode)
    if mode == MODE_OFF:
        return document, {"mode": MODE_OFF, "changed_frames": 0}
    if images is None and frame_paths is None:
        raise ValueError("court paint refinement requires frame images")
    rows = [dict(row) for row in document.get("cameras") or []]
    groups: dict[tuple, list[int]] = {}
    for index, row in enumerate(rows):
        projection = row.get("P")
        if not row.get("supported") or projection is None or row.get("k1") not in (None, 0, 0.0):
            continue
        matrix = np.asarray(projection, float)
        if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
            continue
        groups.setdefault(_group_key(matrix), []).append(index)
    changed = 0
    group_receipts = []
    unread = 0
    refits = 0
    for indexes in sorted(groups.values(), key=len, reverse=True):
        ordered = sorted(indexes, key=lambda item: abs(item - indexes[len(indexes) // 2]))
        tried = []
        chosen = None
        if refits >= MAX_REFITS:
            group_receipts.append(
                {"frames": len(indexes), "applied": False, "tried": [], "budget": True}
            )
            continue
        for index in ordered[:3]:
            if refits >= MAX_REFITS:
                break
            frame = int(rows[index]["frame"])
            image = _load_image(frame, images, frame_paths)
            if image is None:
                continue
            refined, receipt = refine_projection(image, np.asarray(rows[index]["P"], float))
            tried.append({"frame": frame, **receipt})
            if receipt["status"] == "on_paint":
                break
            refits += 1
            if refined is None:
                continue
            confirm = [i for i in ordered if i != index][:1]
            confirmed = True
            for other in confirm:
                other_image = _load_image(int(rows[other]["frame"]), images, frame_paths)
                if other_image is None:
                    continue
                other_gray = cv2.cvtColor(other_image, cv2.COLOR_BGR2GRAY)
                confirmed = (
                    worst_deciding(line_scores(other_gray, ground_homography(refined))) or 0.0
                ) >= ON_PAINT
            if confirmed:
                chosen = refined
                break
        if not tried and refits < MAX_REFITS:
            unread += len(indexes)
        if chosen is not None:
            for index in indexes:
                rows[index][SUPPLIED_P_FIELD] = rows[index]["P"]
                rows[index]["P"] = np.asarray(chosen, float).tolist()
                rows[index]["court_paint_geometry"] = "refit"
                changed += 1
        group_receipts.append(
            {"frames": len(indexes), "applied": chosen is not None, "tried": tried}
        )
    abstained = 0
    run_receipts = []
    if mode == MODE_ABSTAIN:
        for indexes in groups.values():
            for run in _runs(rows, indexes):
                picks = [
                    run[int(i)] for i in np.linspace(0, len(run) - 1, min(VERIFY_FRAMES, len(run)))
                ]
                verdicts = []
                for index in dict.fromkeys(picks):
                    frame = int(rows[index]["frame"])
                    image = _load_image(frame, images, frame_paths)
                    if image is None:
                        unread += 1
                        continue
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    verdicts.append(
                        [frame, frame_on_paint(gray, np.asarray(rows[index]["P"], float))]
                    )
                scored = [ok for _, ok in verdicts if ok is not None]
                off = bool(scored) and sum(scored) * 2 < len(scored)
                if off:
                    for index in run:
                        row = rows[index]
                        supplied = {key: row[key] for key in _ABSTAIN_KEYS if key in row}
                        if SUPPLIED_P_FIELD in row:
                            supplied["P"] = row.pop(SUPPLIED_P_FIELD)
                            changed -= 1
                        row[SUPPLIED_ROW_FIELD] = supplied
                        row.update(
                            supported=False,
                            status="held",
                            reason="court_paint_off",
                            P=None,
                            court_paint_geometry="abstained",
                        )
                        abstained += 1
                run_receipts.append(
                    {
                        "first": int(rows[run[0]]["frame"]),
                        "last": int(rows[run[-1]]["frame"]),
                        "frames": len(run),
                        "abstained": off,
                        "verdicts": verdicts,
                    }
                )
    receipt = {
        "mode": mode,
        "changed_frames": changed,
        "abstained_frames": abstained,
        "unread_frames": unread,
        "refits": refits,
        "groups_over_budget": sum(1 for group in group_receipts if group.get("budget")),
        "groups": group_receipts,
        "runs": run_receipts,
    }
    if not changed and not abstained:
        return document, receipt
    refined_document = dict(document)
    refined_document["cameras"] = rows
    refined_document["court_paint_refinement"] = mode
    return refined_document, receipt
