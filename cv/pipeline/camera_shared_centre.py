"""One camera centre per match: the vertical column of every frame's projection.

A ground homography fixes a projection's three ground columns; the fourth (vertical) column is
then fixed by the camera centre ``C`` alone, because ``P @ [C, 1] = 0``::

    p3 = -(h1 * Cx + h2 * Cy + h4) / Cz

The broadcast play camera pans, tilts and zooms on a fixed mount, so one ``C`` serves the whole
match. Per-point calibration instead solved focal from the nearly flat net tape, which pins the
camera's elevation seen from the net but not its distance: the solved centre wandered between
points (16-28 m behind the baseline within one match) and heights away from the net were scaled
wrongly in opposite directions near and far (cv/experiments/camera_height/REPORT.md).

``fit_shared_centre`` chooses ``C`` from two automatic cues:

- every view's implied intrinsics are physical (square pixels, zero skew, principal point at the
  picture centre). One telephoto view constrains this poorly; hundreds sharing a centre do not,
  except along the camera's viewing distance, where it stays slightly short;
- player boxes balance: both players play both ends, so over a match the median metric height
  of near-side boxes equals that of far-side boxes. This pins the distance.

The intrinsics alone put the camera ~10% short of where Hawk-Eye heights put it; there the
implied principal point stays at the picture centre, but it moves by hundreds of pixels within a
few metres of distance, so the ground homographies carry a small perspective bias and the
intrinsics cannot arbitrate distance. They are used to find the camera's direction and to reject
matches no shared square-pixel camera explains; the box balance then sets the distance, bounded
to ``MAXIMUM_BOX_DISTANCE_CHANGE`` of the intrinsic solution.

Ground projection is unchanged by construction (court-plane positions, bounces and player roots
are untouched). A match whose views stay inconsistent with any shared centre (another camera,
an anamorphic picture) abstains and keeps its per-point cameras.

``TENNIS_CAMERA_SHARED_CENTRE=on`` enables the ``camera_shared_centre`` stage (default off).

    python -m cv.pipeline.camera_shared_centre --match-out OUT/upstream/MATCH
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ENV = "TENNIS_CAMERA_SHARED_CENTRE"
POINT_CAMERA = "camera_P_per_frame_point_v1.npz"
CAMERA = "camera_P_per_frame_v1.npz"
RECEIPT = "camera_shared_centre_v1.json"
SCHEMA = "camera_shared_centre_v1"
IMAGE_SIZE = (1920, 1080)

INTRINSIC_RESIDUAL_SCALE = 0.02
MAXIMUM_FIT_VIEWS = 600
MAXIMUM_BOX_ROWS = 4000
MINIMUM_BOX_ROWS_PER_SIDE = 200
MINIMUM_BOX_CONFIDENCE = 0.6
# The balance is one number per match against ~2400 intrinsic residuals; this weight makes it
# decide the viewing distance while the intrinsics keep the direction (chosen on development
# matches, see the report).
BOX_BALANCE_WEIGHT = 300.0
# A view whose implied intrinsics stay this far from physical under the shared centre is
# reported as inconsistent (a bad ground solve, a replay angle).
MAXIMUM_VIEW_INTRINSIC_RESIDUAL = 0.05
# Abstain when, at the intrinsic-only solution, the median view is this far from physical
# (focal units), or when the box balance moves the camera further than this share of its
# distance from the court centre.
MAXIMUM_MEDIAN_ASPECT_RESIDUAL = 0.02
MAXIMUM_MEDIAN_PRINCIPAL_RESIDUAL = 0.05
MINIMUM_CONSISTENT_SHARE = 0.6
MAXIMUM_BOX_DISTANCE_CHANGE = 0.20
COURT_CENTRE = np.array([5.485, 11.885, 0.0])


def enabled() -> bool:
    return os.environ.get(ENV, "off").strip().lower() in {"1", "on", "true"}


def normalised_ground(projections: np.ndarray) -> np.ndarray:
    """``(n, 3, 3)`` court-to-image ground columns with unit Frobenius norm."""
    projections = np.asarray(projections, dtype=float).reshape(-1, 3, 4)
    ground = projections[:, :, [0, 1, 3]]
    return ground / np.linalg.norm(ground, axis=(1, 2), keepdims=True)


def projections_with_centre(projections: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Keep each projection's ground columns and set its vertical column from ``centre``."""
    ground = normalised_ground(projections)
    cx, cy, cz = (float(v) for v in centre)
    vertical = -(ground[:, :, 0] * cx + ground[:, :, 1] * cy + ground[:, :, 2]) / cz
    return np.stack((ground[:, :, 0], ground[:, :, 1], vertical, ground[:, :, 2]), axis=2)


def implied_intrinsics(projections: np.ndarray) -> np.ndarray:
    """``(n, 5)`` fx, fy, skew, cx, cy of each projection's left 3x3 (closed-form RQ)."""
    m = np.asarray(projections, dtype=float)[:, :, :3]
    a = m @ np.swapaxes(m, 1, 2)
    a = a / a[:, 2:3, 2:3]
    cx, cy = a[:, 0, 2], a[:, 1, 2]
    fy = np.sqrt(np.maximum(a[:, 1, 1] - cy**2, 1e-12))
    skew = (a[:, 0, 1] - cx * cy) / fy
    fx = np.sqrt(np.maximum(a[:, 0, 0] - skew**2 - cx**2, 1e-12))
    return np.stack((fx, fy, skew, cx, cy), axis=1)


def intrinsic_residuals(
    projections: np.ndarray, centre: np.ndarray, image_size: tuple[int, int] = IMAGE_SIZE
) -> np.ndarray:
    """``(n, 4)`` aspect-1, skew, and principal-point offsets (all in focal units)."""
    width, height = image_size
    k = implied_intrinsics(projections_with_centre(projections, centre))
    fx = k[:, 0]
    return np.stack(
        (
            k[:, 1] / fx - 1.0,
            k[:, 2] / fx,
            (k[:, 3] - width / 2.0) / fx,
            (k[:, 4] - height / 2.0) / fx,
        ),
        axis=1,
    )


def camera_centre(projection: np.ndarray) -> np.ndarray:
    _, _, vt = np.linalg.svd(np.asarray(projection, dtype=float))
    c = vt[-1]
    return c[:3] / c[3]


@dataclass(frozen=True)
class PlayerBoxes:
    """Sided player boxes: per row its frame's projection, court root, pixel height, side."""

    projections: np.ndarray  # (n, 3, 4)
    roots: np.ndarray  # (n, 2) court x, y
    pixel_heights: np.ndarray  # (n,)
    far: np.ndarray  # (n,) bool


def metric_heights(projections: np.ndarray, boxes: PlayerBoxes) -> np.ndarray:
    """Height above the box root whose image lies the box's pixel height above the root."""
    ground = np.c_[boxes.roots, np.zeros(len(boxes.roots)), np.ones(len(boxes.roots))]
    a = np.einsum("nij,nj->ni", projections, ground)
    b = projections[:, :, 2]
    top = a[:, 1] / a[:, 2] - boxes.pixel_heights
    # (a_v + h b_v) / (a_w + h b_w) = top  ->  h = (a_v - top a_w) / (top b_w - b_v)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a[:, 1] - top * a[:, 2]) / (top * b[:, 2] - b[:, 1])


def _interquartile_mean(values: np.ndarray) -> float:
    values = np.sort(values[np.isfinite(values)])
    n = len(values)
    return float(values[n // 4 : max(n // 4 + 1, 3 * n // 4)].mean())


def box_log_ratio(centre: np.ndarray, boxes: PlayerBoxes) -> float:
    """log(near / far) of interquartile-mean metric box heights under ``centre``."""
    heights = metric_heights(projections_with_centre(boxes.projections, centre), boxes)
    with np.errstate(invalid="ignore", divide="ignore"):
        logs = np.log(heights)
    return _interquartile_mean(logs[~boxes.far]) - _interquartile_mean(logs[boxes.far])


@dataclass(frozen=True)
class SharedCentre:
    centre: np.ndarray
    intrinsic_centre: np.ndarray
    views: int
    consistent_views: int  # at the intrinsic-only solution
    median_residual: np.ndarray  # |aspect-1|, |skew|, |cx|, |cy| at the intrinsic solution
    focal_px: np.ndarray  # p10, p50, p90 of implied fx at the final centre
    box_rows: int
    box_log_ratio_intrinsic: float | None
    box_log_ratio: float | None
    distance_change: float  # final / intrinsic distance from the court centre, minus 1
    success: bool

    def accepted(self) -> tuple[bool, str]:
        if not self.success:
            return False, "fit_failed"
        if self.median_residual[0] > MAXIMUM_MEDIAN_ASPECT_RESIDUAL:
            return False, "views_not_square_pixel"
        if self.median_residual[3] > MAXIMUM_MEDIAN_PRINCIPAL_RESIDUAL:
            return False, "views_not_centred"
        if self.consistent_views < MINIMUM_CONSISTENT_SHARE * self.views:
            return False, "too_few_consistent_views"
        if abs(self.distance_change) > MAXIMUM_BOX_DISTANCE_CHANGE:
            return False, "box_balance_moved_camera_too_far"
        return True, "accepted"


def fit_shared_centre(
    projections: np.ndarray,
    image_size: tuple[int, int] = IMAGE_SIZE,
    boxes: PlayerBoxes | None = None,
    initial: np.ndarray | None = None,
) -> SharedCentre:
    """Robust least-squares camera centre over ``projections`` (one or more per view)."""
    projections = np.asarray(projections, dtype=float).reshape(-1, 3, 4)
    projections = projections[np.isfinite(projections).all(axis=(1, 2))]
    if len(projections) > MAXIMUM_FIT_VIEWS:
        keep = np.linspace(0, len(projections) - 1, MAXIMUM_FIT_VIEWS).round().astype(int)
        projections = projections[keep]
    if len(projections) < 3:
        raise ValueError("at least three views are needed for a shared camera centre")
    if initial is None:
        centres = np.asarray([camera_centre(p) for p in projections])
        initial = np.median(centres[np.isfinite(centres).all(axis=1)], axis=0)
        if not np.isfinite(initial).all() or initial[2] <= 0.5:
            initial = np.array([5.485, -25.0, 9.0])
    use_boxes = (
        boxes is not None
        and int(np.sum(boxes.far)) >= MINIMUM_BOX_ROWS_PER_SIDE
        and int(np.sum(~boxes.far)) >= MINIMUM_BOX_ROWS_PER_SIDE
    )

    def residuals(centre):
        values = intrinsic_residuals(projections, centre, image_size).ravel()
        if use_boxes:
            balance = box_log_ratio(centre, boxes)
            weight = BOX_BALANCE_WEIGHT * np.sqrt(len(values))
            values = np.append(values, weight * (balance if np.isfinite(balance) else 1.0))
        return values

    first = least_squares(
        lambda c: intrinsic_residuals(projections, c, image_size).ravel(),
        np.asarray(initial, dtype=float),
        loss="soft_l1",
        f_scale=INTRINSIC_RESIDUAL_SCALE,
    )
    fit = (
        least_squares(residuals, first.x, loss="soft_l1", f_scale=INTRINSIC_RESIDUAL_SCALE)
        if use_boxes
        else first
    )
    centre = fit.x
    intrinsic = intrinsic_residuals(projections, first.x, image_size)
    worst = np.max(np.abs(intrinsic), axis=1)
    k = implied_intrinsics(projections_with_centre(projections, centre))
    change = float(
        np.linalg.norm(centre - COURT_CENTRE) / np.linalg.norm(first.x - COURT_CENTRE) - 1.0
    )
    return SharedCentre(
        centre=centre,
        intrinsic_centre=first.x,
        views=len(projections),
        consistent_views=int(np.sum(worst <= MAXIMUM_VIEW_INTRINSIC_RESIDUAL)),
        median_residual=np.median(np.abs(intrinsic), axis=0),
        focal_px=np.percentile(k[:, 0], [10, 50, 90]),
        box_rows=int(len(boxes.far)) if use_boxes else 0,
        box_log_ratio_intrinsic=float(box_log_ratio(first.x, boxes)) if use_boxes else None,
        box_log_ratio=float(box_log_ratio(centre, boxes)) if use_boxes else None,
        distance_change=change,
        success=bool(first.success and fit.success and centre[2] > 0.5),
    )


def _frame_number(value: str) -> int:
    return int(Path(value).stem.rsplit("_", 1)[-1])


def load_player_boxes(path: Path, cameras, maximum: int = MAXIMUM_BOX_ROWS) -> PlayerBoxes | None:
    """Confident sided boxes (court roots in a band around the court) on camera frames."""
    if not path.is_file():
        return None
    index = {
        (str(c), int(f)): i for i, (c, f) in enumerate(zip(cameras["clips"], cameras["frames"]))
    }
    rows, frames, far = [], [], []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("side") not in {"near", "far"} or not row.get("court_y"):
                continue
            if float(row.get("conf") or 0.0) < MINIMUM_BOX_CONFIDENCE:
                continue
            court_x, court_y = float(row["court_x"]), float(row["court_y"])
            if not (-6.0 < court_y < 30.0 and -4.0 < court_x < 15.0):
                continue
            camera_row = index.get((row["clip"], _frame_number(row["frame"])))
            if camera_row is None:
                continue
            height = float(row["y1_native"]) - float(row["y0_native"])
            if height <= 0:
                continue
            rows.append((court_x, court_y, height))
            frames.append(camera_row)
            far.append(row["side"] == "far")
    if not rows:
        return None
    keep = np.linspace(0, len(rows) - 1, min(len(rows), maximum)).round().astype(int)
    rows_np = np.asarray(rows, dtype=float)[keep]
    return PlayerBoxes(
        projections=np.asarray(cameras["P"], dtype=float)[np.asarray(frames)[keep]],
        roots=rows_np[:, :2],
        pixel_heights=rows_np[:, 2],
        far=np.asarray(far, dtype=bool)[keep],
    )


def sided_boxes_path(match_out: Path) -> Path | None:
    found = sorted(Path(match_out).glob("player_boxes_*_native_sided_v1.csv"))
    return found[0] if found else None


def apply_match(
    match_out: Path,
    source_name: str = POINT_CAMERA,
    output_name: str = CAMERA,
    boxes_path: Path | None = None,
) -> dict:
    """Write ``output_name`` with shared-centre vertical columns (or a copy, on abstention)."""
    match_out = Path(match_out)
    source = match_out / source_name
    output = match_out / output_name
    with np.load(source, allow_pickle=False) as data:
        cameras = {key: data[key] for key in data.files}
    receipt: dict = {
        "schema": SCHEMA,
        "automatic": True,
        "switch": f"{ENV}=on",
        "source": source_name,
        "output": output_name,
        "image_size": list(IMAGE_SIZE),
    }
    projections = np.asarray(cameras["P"], dtype=float)
    clips = np.asarray(cameras["clips"]).astype(str)
    status, reason, solution = "abstained", "no_views", None
    if len(projections):
        reliable = np.asarray(cameras.get("reliable", np.ones(len(clips), bool)), dtype=bool)
        views = [np.flatnonzero((clips == clip) & reliable) for clip in np.unique(clips)]
        first = np.asarray([rows[0] for rows in views if len(rows)], dtype=int)
        boxes_path = boxes_path or sided_boxes_path(match_out)
        boxes = load_player_boxes(boxes_path, cameras) if boxes_path else None
        if len(first) >= 3:
            solution = fit_shared_centre(projections[first], IMAGE_SIZE, boxes=boxes)
            accepted, reason = solution.accepted()
            status = "applied" if accepted else "abstained"
            old_centres = np.asarray([camera_centre(projections[i]) for i in first])
            receipt["point_camera_centre_p10_50_90"] = (
                np.percentile(old_centres, [10, 50, 90], axis=0).round(3).tolist()
            )
        else:
            reason = "too_few_reliable_views"
        receipt["boxes"] = os.fspath(boxes_path) if boxes_path else None
    if solution is not None:
        receipt["solution"] = {
            key: (value.round(5).tolist() if isinstance(value, np.ndarray) else value)
            for key, value in asdict(solution).items()
        }
    receipt["status"] = status
    receipt["reason"] = reason
    if status == "applied":
        cameras["P"] = projections_with_centre(projections, solution.centre)
        temporary = output.with_name(output.name + ".tmp.npz")
        np.savez_compressed(temporary, **cameras)
        temporary.replace(output)
    elif source != output:
        shutil.copyfile(source, output)
    (match_out / RECEIPT).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--match-out", type=Path, required=True)
    parser.add_argument("--source", default=POINT_CAMERA)
    parser.add_argument("--output", default=CAMERA)
    args = parser.parse_args()
    receipt = apply_match(args.match_out, args.source, args.output)
    print(receipt["status"], receipt["reason"], (receipt.get("solution") or {}).get("centre"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
