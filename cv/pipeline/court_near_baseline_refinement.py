"""Put an intrinsic-fallback camera's near baseline back on the paint.

The ground homography used by ``intrinsic_fallback`` is fixed by the far court. On
the Rome broadcast that leaves the near service line and the near baseline tens to
hundreds of pixels below the paint, while the far lines stay on it. Reprojection
against the homography cannot see the error. This module searches the native frame
for those two near lines and refits the projection. It does not read any clicked
court reference. When the paint is visible and the refit would move the far lines,
the frame is marked unresolved instead of keeping the bad near-half geometry.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from cv.pipeline import court
from cv.pipeline.camera_artifacts import fallback_point_camera_source
from cv.pipeline.camera_cal import intrinsic_projection_from_ground

MODE_OFF = "off"
MODE_ON = "on"
MODES = (MODE_OFF, MODE_ON)

_SERVICE_Y = court.NET_Y - court.SERVICE_FROM_NET
_FAR_SERVICE_Y = court.NET_Y + court.SERVICE_FROM_NET
_HORIZONTALS = (
    ("near_baseline", 0.0, "near"),
    ("near_service", _SERVICE_Y, "near"),
    ("far_service", _FAR_SERVICE_Y, "far"),
    ("far_baseline", court.COURT_L, "far"),
)
_SIDELINES = (
    ("left_doubles", 0.0),
    ("right_doubles", court.COURT_W),
)


def validate_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError("explicit near-baseline refinement mode required")
    return mode


def _project(projection: np.ndarray, court_xy: tuple[float, float]) -> np.ndarray:
    point = np.array([court_xy[0], court_xy[1], 0.0, 1.0], dtype=float)
    mapped = projection @ point
    return mapped[:2] / mapped[2]


def _line_through(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    direction = np.asarray(end, float) - np.asarray(start, float)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    length = float(np.linalg.norm(normal))
    if length < 1e-8:
        raise ValueError("degenerate court line")
    normal /= length
    return np.array([normal[0], normal[1], -float(np.dot(normal, start))], dtype=float)


def _intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    point = np.cross(first, second)
    if abs(float(point[2])) < 1e-8:
        return None
    return point[:2] / point[2]


def _toward_net(projection: np.ndarray) -> np.ndarray:
    near = _project(projection, (court.COURT_W / 2.0, 0.0))
    far = _project(projection, (court.COURT_W / 2.0, court.COURT_L))
    direction = far - near
    length = float(np.linalg.norm(direction))
    if length < 1.0:
        return np.array([0.0, -1.0])
    return direction / length


def _tophat(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = max(15, int(round(image.shape[1] / 96.0)))
    if kernel % 2 == 0:
        kernel += 1
    hat = cv2.morphologyEx(
        gray,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel)),
    )
    return gray, hat


def _sample_line(
    gray: np.ndarray,
    hat: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    normal: np.ndarray,
    *,
    near_offset: float,
    far_offset: float,
    step: float,
) -> tuple[np.ndarray, float]:
    """Return inlier paint pixels and the median shift along ``normal``.

    ``normal`` points toward the net. Offsets run from ``near_offset`` (negative:
    away from the net) to ``far_offset``.
    """
    samples = np.linspace(start, end, 48)[4:-4]
    offsets = np.arange(near_offset, far_offset + 0.5 * step, step)
    if len(samples) < 8 or len(offsets) < 3:
        return np.empty((0, 2)), 0.0
    chosen = []
    shifts = []
    for point in samples:
        coords = point + offsets[:, None] * normal
        xs = np.rint(coords[:, 0]).astype(int)
        ys = np.rint(coords[:, 1]).astype(int)
        valid = (xs >= 1) & (ys >= 1) & (xs < gray.shape[1] - 1) & (ys < gray.shape[0] - 1)
        if int(valid.sum()) < 5:
            continue
        response = np.zeros(len(offsets), dtype=float)
        response[valid] = hat[ys[valid], xs[valid]]
        index = int(np.argmax(response))
        if not valid[index]:
            continue
        baseline = float(np.median(response[valid]))
        prominence = float(response[index] - baseline)
        if prominence < 18.0 or float(gray[ys[index], xs[index]]) < 145.0:
            continue
        chosen.append(coords[index])
        shifts.append(float(offsets[index]))
    if len(chosen) < 8:
        return np.empty((0, 2)), 0.0
    # The seed line can be tilted relative to the paint, so the offset is not
    # constant along it. Fit the paint, then drop points off that line.
    cloud = np.asarray(chosen, dtype=float)
    fitted = _fit_line(cloud)
    if fitted is None:
        return np.empty((0, 2)), 0.0
    distance = np.abs(cloud @ fitted[:2] + fitted[2])
    kept_index = distance <= 4.0
    if int(kept_index.sum()) < max(8, int(0.35 * len(samples))):
        return np.empty((0, 2)), float(np.median(shifts))
    kept_shifts = [value for value, keep in zip(shifts, kept_index) if keep]
    return cloud[kept_index], float(np.median(kept_shifts))


def _fit_line(points: np.ndarray) -> np.ndarray | None:
    if len(points) < 2:
        return None
    fit = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    direction = fit[:2]
    origin = fit[2:]
    normal = np.array([-direction[1], direction[0]], dtype=float)
    length = float(np.linalg.norm(normal))
    if length < 1e-8:
        return None
    normal /= length
    return np.array([normal[0], normal[1], -float(np.dot(normal, origin))], dtype=float)


def _model_segment(
    projection: np.ndarray, court_y: float | None, court_x: float | None
) -> tuple[np.ndarray, np.ndarray]:
    if court_y is not None:
        return (
            _project(projection, (0.0, court_y)),
            _project(projection, (court.COURT_W, court_y)),
        )
    return (
        _project(projection, (court_x, 0.0)),
        _project(projection, (court_x, court.COURT_L)),
    )


def _line_distance(line: np.ndarray, points: np.ndarray) -> float:
    normal = line[:2]
    return float(np.median(np.abs(points @ normal + line[2])))


def refine_projection(
    image: np.ndarray,
    projection: np.ndarray,
) -> tuple[np.ndarray | None, dict]:
    """Return a corrected projection, or None when the near half stays unresolved.

    ``image`` is a native BGR frame. ``projection`` maps court metres to that
    frame. The returned projection keeps the far lines and moves the near
    baseline and near service line onto the paint that is actually in the frame.
    """
    if image.ndim != 3 or projection.shape != (3, 4) or not np.isfinite(projection).all():
        return None, {"status": "unresolved", "reason": "bad_input"}
    height, width = image.shape[:2]
    gray, hat = _tophat(image)
    toward = _toward_net(projection)
    wide = 0.22 * height
    narrow = max(6.0, 0.008 * height)
    lines: dict[str, np.ndarray] = {}
    evidence: dict[str, dict] = {}

    def search(
        name: str, start: np.ndarray, end: np.ndarray, reach: float, *, toward_net: bool
    ) -> None:
        if toward_net:
            normal = toward
            near_offset = -min(24.0, 0.025 * height)
            far_offset = reach
        else:
            direction = np.asarray(end, float) - np.asarray(start, float)
            side = np.array([-direction[1], direction[0]], dtype=float)
            length = float(np.linalg.norm(side))
            normal = side / length if length else toward
            near_offset = -reach
            far_offset = reach
        found, shift = _sample_line(
            gray,
            hat,
            start,
            end,
            normal,
            near_offset=near_offset,
            far_offset=far_offset,
            step=1.0,
        )
        fitted = _fit_line(found)
        # Far lines and sidelines are already on the paint. A large shift there is
        # a neighbouring line, not a refinement, so the model line stays.
        if (
            fitted is None
            or (not toward_net and abs(shift) > 5.0)
            or (name in {"far_service", "far_baseline"} and abs(shift) > 5.0)
        ):
            lines[name] = _line_through(start, end)
            evidence[name] = {
                "detected": False,
                "shift_px": None if fitted is None else round(shift, 2),
            }
            return
        lines[name] = fitted
        evidence[name] = {
            "detected": True,
            "shift_px": round(shift, 2),
            "inliers": int(len(found)),
            "residual_px": round(_line_distance(fitted, found), 2),
        }

    near_base = _model_segment(projection, 0.0, None)
    near_service = _model_segment(projection, _SERVICE_Y, None)
    far_service = _model_segment(projection, _FAR_SERVICE_Y, None)
    gap_to_service = float(np.linalg.norm(near_service[0] - near_base[0]))
    gap_to_far = float(np.linalg.norm(far_service[0] - near_service[0]))

    def search_near(name: str, segment: tuple[np.ndarray, np.ndarray], reach: float) -> None:
        # A line already next to the seed must win before a wide search can lock
        # onto a brighter line further up the court.
        search(name, *segment, min(32.0, reach), toward_net=True)
        if evidence.get(name, {}).get("detected"):
            return
        search(name, *segment, reach, toward_net=True)

    search_near("near_baseline", near_base, min(wide, 0.72 * gap_to_service))
    search_near("near_service", near_service, min(0.12 * height, 0.60 * gap_to_far))
    for name, court_y in (("far_service", _FAR_SERVICE_Y), ("far_baseline", court.COURT_L)):
        search(name, *_model_segment(projection, court_y, None), narrow, toward_net=True)
    for name, court_x in _SIDELINES:
        start, end = _model_segment(projection, None, court_x)
        # The far end of each sideline is already on the paint. The near end is
        # what a bad fallback swings away from the doubles line.
        start = start + 0.84 * (end - start)
        search(name, start, end, max(narrow, 12.0), toward_net=False)

    if not evidence["near_baseline"]["detected"]:
        return None, {
            "status": "unresolved",
            "reason": "near_baseline_not_visible",
            "lines": evidence,
        }
    if not evidence["near_service"]["detected"]:
        return None, {
            "status": "unresolved",
            "reason": "near_service_not_visible",
            "lines": evidence,
        }

    # Four corners only. The far baseline and the two sidelines stay put, and
    # the detected near baseline is the fourth constraint. Extra service-line
    # points would drag the far court off the paint it already matches.
    image_points = []
    court_points = []
    for horizontal, court_y in (("far_baseline", court.COURT_L), ("near_baseline", 0.0)):
        for vertical, court_x in _SIDELINES:
            point = _intersection(lines[horizontal], lines[vertical])
            if point is None or not np.isfinite(point).all():
                return None, {"status": "unresolved", "reason": "parallel_lines", "lines": evidence}
            image_points.append(point)
            court_points.append((court_x, court_y))
    court_to_image = cv2.getPerspectiveTransform(
        np.asarray(court_points, np.float32),
        np.asarray(image_points, np.float32),
    )
    if court_to_image is None or not np.isfinite(court_to_image).all():
        return None, {"status": "unresolved", "reason": "homography_failed", "lines": evidence}
    solved = intrinsic_projection_from_ground(court_to_image, w=width, h=height)
    if solved is None:
        return None, {"status": "unresolved", "reason": "intrinsic_solve_failed", "lines": evidence}
    refined, focal = solved
    far_move = []
    for court_y in (_FAR_SERVICE_Y, court.COURT_L):
        before = _project(projection, (court.COURT_W / 2.0, court_y))
        after = _project(refined, (court.COURT_W / 2.0, court_y))
        far_move.append(float(np.linalg.norm(after - before)))
    near_points, _ = _sample_line(
        gray,
        hat,
        *_model_segment(refined, 0.0, None),
        toward,
        near_offset=-8.0,
        far_offset=8.0,
        step=1.0,
    )
    near_residual = (
        _line_distance(_line_through(*_model_segment(refined, 0.0, None)), near_points)
        if len(near_points)
        else 99.0
    )
    receipt = {
        "status": "resolved",
        "reason": "near_lines_on_paint",
        "focal_px": round(float(focal), 1),
        "far_line_move_px": round(max(far_move), 2),
        "near_baseline_residual_px": round(near_residual, 2),
        "lines": evidence,
    }
    if max(far_move) > 8.0 or near_residual > 4.0:
        receipt["status"] = "unresolved"
        receipt["reason"] = (
            "refit_moved_far_lines" if max(far_move) > 8.0 else "near_baseline_not_on_paint"
        )
        return None, receipt
    return refined, receipt


def _group_key(projection: np.ndarray) -> tuple[int, int, int, int]:
    left = _project(projection, (0.0, 0.0))
    right = _project(projection, (court.COURT_W, 0.0))
    return (
        int(round(left[0] / 8.0)),
        int(round(left[1] / 4.0)),
        int(round(right[0] / 8.0)),
        int(round(right[1] / 4.0)),
    )


def _existing_frame_path(path: str) -> str | None:
    """Return a readable path, rewriting a foreign ``/tennis-data/`` prefix onto this machine.

    A relative path is a ``TENNIS_DATA_ROOT``-based file record, independent of the cwd.
    """
    if Path(path).is_file():
        return path
    root = os.environ.get("TENNIS_DATA_ROOT")
    if root and not Path(path).is_absolute() and (Path(root) / path).is_file():
        return str(Path(root) / path)
    marker = "/tennis-data/"
    if root and marker in path:
        rewritten = Path(root) / path.split(marker, 1)[1]
        if rewritten.is_file():
            return str(rewritten)
    return None


def _load_image(
    frame: int, images: Mapping[int, np.ndarray] | None, frame_paths: Mapping[int, str] | None
) -> np.ndarray | None:
    if images is not None and frame in images:
        return images[frame]
    path = None if frame_paths is None else frame_paths.get(frame)
    if not path:
        return None
    resolved = _existing_frame_path(str(path))
    if resolved is None:
        return None
    return cv2.imread(resolved)


_ROW_FIELDS_REFINEMENT_MAY_CHANGE = frozenset(
    {"P", "supported", "status", "reason", "near_half_geometry"}
)


def same_component_source(supplied: dict, bound: dict) -> bool:
    """True when ``supplied`` is ``bound`` or a near-baseline refinement of it.

    The component plan is hashed against the camera document it was prepared
    from. The fit has to project through the refined document, so the two are
    allowed to differ only on intrinsic-fallback rows, and only by the fields
    ``refine_camera_document`` writes. Any other change is still a different source.
    """
    if supplied == bound:
        return True
    if supplied.get("near_baseline_refinement") != MODE_ON:
        return False
    if bound.get("near_baseline_refinement") not in (None, MODE_OFF):
        return False
    if {
        key: value
        for key, value in supplied.items()
        if key not in {"cameras", "near_baseline_refinement"}
    } != {
        key: value
        for key, value in bound.items()
        if key not in {"cameras", "near_baseline_refinement"}
    }:
        return False
    supplied_rows = supplied.get("cameras") or []
    bound_rows = bound.get("cameras") or []
    if len(supplied_rows) != len(bound_rows):
        return False
    changed = False
    for new, old in zip(supplied_rows, bound_rows):
        if new == old:
            continue
        changed = True
        source = str(old.get("source") or "")
        if not (
            old.get("supported")
            and fallback_point_camera_source(source)
            and old.get("k1") in (None, 0, 0.0)
            and old.get("P") is not None
        ):
            return False
        if {
            key: value for key, value in new.items() if key not in _ROW_FIELDS_REFINEMENT_MAY_CHANGE
        } != {
            key: value for key, value in old.items() if key not in _ROW_FIELDS_REFINEMENT_MAY_CHANGE
        }:
            return False
        geometry = new.get("near_half_geometry")
        if geometry == "resolved":
            if (
                new.get("supported") is not True
                or not isinstance(new.get("P"), list)
                or new.get("status") != old.get("status")
                or new.get("reason") != old.get("reason")
            ):
                return False
        elif geometry == "unresolved":
            if not (
                new.get("supported") is False
                and new.get("P") is None
                and new.get("status") == "held"
                and new.get("reason") == "near_half_geometry_unresolved"
            ):
                return False
        else:
            return False
    return changed


def refine_camera_document(
    document: dict,
    images: Mapping[int, np.ndarray] | None = None,
    *,
    mode: str,
    frame_paths: Mapping[int, str] | None = None,
) -> tuple[dict, dict]:
    """Refine supported intrinsic-fallback rows. ``off`` returns the document unchanged."""
    validate_mode(mode)
    if mode == MODE_OFF:
        return document, {"mode": MODE_OFF, "changed_frames": 0, "unresolved_frames": 0}
    if images is None and frame_paths is None:
        raise ValueError("near-baseline refinement requires frame images")
    rows = []
    groups: dict[tuple, list[int]] = {}
    for index, row in enumerate(document.get("cameras") or []):
        copied = dict(row)
        rows.append(copied)
        source = str(copied.get("source") or "")
        projection = copied.get("P")
        if (
            not copied.get("supported")
            or not fallback_point_camera_source(source)
            or projection is None
        ):
            continue
        matrix = np.asarray(projection, float)
        if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
            continue
        if copied.get("k1") not in (None, 0, 0.0):
            continue
        groups.setdefault(_group_key(matrix), []).append(index)

    changed = 0
    unresolved = 0
    group_receipts = []
    for key, indexes in groups.items():
        ordered = sorted(indexes, key=lambda item: abs(item - indexes[len(indexes) // 2]))
        chosen = None
        receipt = {"status": "unresolved", "reason": "image_unavailable"}
        for index in ordered[:8]:
            frame = int(rows[index]["frame"])
            image = _load_image(frame, images, frame_paths)
            if image is None:
                continue
            refined, receipt = refine_projection(image, np.asarray(rows[index]["P"], float))
            if refined is not None:
                chosen = refined
                break
        if chosen is None and receipt.get("reason") == "image_unavailable":
            group_receipts.append({"frames": len(indexes), **receipt})
            continue
        for index in indexes:
            if chosen is None:
                rows[index]["supported"] = False
                rows[index]["status"] = "held"
                rows[index]["reason"] = "near_half_geometry_unresolved"
                rows[index]["P"] = None
                rows[index]["near_half_geometry"] = "unresolved"
                unresolved += 1
            else:
                rows[index]["P"] = np.asarray(chosen, float).tolist()
                rows[index]["near_half_geometry"] = "resolved"
                changed += 1
        group_receipts.append({"frames": len(indexes), "applied": chosen is not None, **receipt})
    refined_document = dict(document)
    refined_document["cameras"] = rows
    refined_document["near_baseline_refinement"] = mode
    return refined_document, {
        "mode": mode,
        "changed_frames": changed,
        "unresolved_frames": unresolved,
        "groups": group_receipts,
    }
