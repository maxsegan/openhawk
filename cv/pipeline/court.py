"""Court homography for a static play camera — Phase B.

The behind-baseline play camera barely moves, so ONE homography per match maps image
coordinates to real court coordinates. We estimate it from the court's white lines on a
median play frame (robust to players/ball), then use it to (a) filter person boxes to the
two competitors and (b) express positions in court meters (direction/depth geometry).

Court frame: x across the baseline (0..10.97m doubles), y along the court (0..23.77m),
origin at the near court's left doubles corner viewed from the camera.

Approach: white-line mask -> Hough segments -> split into ~horizontal / ~vertical families
-> take extreme horizontals (near/far baselines) and extreme verticals (left/right doubles
sidelines) -> their 4 intersections + known court size -> cv2.getPerspectiveTransform.
Validated by reprojecting the service lines and center line onto the frame.

    .venv/bin/python cv/pipeline/court.py --out data/processed/rg2025f
"""

from __future__ import annotations

import argparse
import glob
import heapq
import os

import cv2
import numpy as np

from cv.pipeline import resolution as res
from cv.pipeline.court_geometry_gate import assess_court_homography

COURT_W = 10.97  # doubles width (m)
COURT_L = 23.77  # length (m)
SERVICE_FROM_NET = 6.40
NET_Y = COURT_L / 2


def _image_size(image: np.ndarray) -> res.FrameSize:
    height, width = image.shape[:2]
    return res.FrameSize(width, height)


def _scaled_px(value: float, image: np.ndarray, *, odd: bool = False) -> int:
    result = max(1, int(round(res.pixel_length(value, res.CANONICAL_SIZE, _image_size(image)))))
    if odd and result % 2 == 0:
        result += 1
    return result


def median_play_frame(out_dir: str, n: int = 60) -> np.ndarray:
    frames = sorted(glob.glob(os.path.join(out_dir, "rally_frames", "pt*", "f_*.jpg")))
    if not frames:
        raise SystemExit("no rally frames; run players.py first")
    step = max(1, len(frames) // n)
    imgs = [cv2.imread(f) for f in frames[::step][:n]]
    return np.median(np.stack([i for i in imgs if i is not None]), axis=0).astype(np.uint8)


def line_mask_observation_mask(shape: tuple[int, ...], top_fraction: float = 0.22) -> np.ndarray:
    """Pixels retained by the court-line extractor, before testing for painted lines."""
    h, w = shape[:2]
    observed = np.ones((h, w), dtype=bool)
    observed[: int(h * top_fraction)] = False
    observed[:, : int(w * 0.06)] = False
    observed[:, int(w * 0.94) :] = False
    observed[int(h * 0.84) :, : int(w * 0.45)] = False
    return observed


def line_mask(img: np.ndarray, top_fraction: float = 0.22) -> np.ndarray:
    """Thin bright low-saturation structures via top-hat (robust to faint lines on clay).
    Crowd band (top), stand margins (sides) and scoreboard (bottom-left) are zeroed."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    kernel = _scaled_px(15, img, odd=True)
    th = cv2.morphologyEx(
        gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    )
    mask = ((th > 38) & (hsv[..., 1] < 90)).astype(np.uint8) * 255
    mask[~line_mask_observation_mask(mask.shape, top_fraction)] = 0
    return mask


def _intersect(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
    return px, py


COURT_MODEL_LINES = None  # filled below (all standard court lines in court coords)


def _court_model():
    global COURT_MODEL_LINES
    if COURT_MODEL_LINES is None:
        sw = (COURT_W - 8.23) / 2  # singles sideline inset (8.23m singles width)
        COURT_MODEL_LINES = [
            [(0, 0), (COURT_W, 0)],
            [(0, COURT_L), (COURT_W, COURT_L)],
            [(0, 0), (0, COURT_L)],
            [(COURT_W, 0), (COURT_W, COURT_L)],
            [(sw, 0), (sw, COURT_L)],
            [(COURT_W - sw, 0), (COURT_W - sw, COURT_L)],
            [(0, NET_Y - SERVICE_FROM_NET), (COURT_W, NET_Y - SERVICE_FROM_NET)],
            [(0, NET_Y + SERVICE_FROM_NET), (COURT_W, NET_Y + SERVICE_FROM_NET)],
            [(COURT_W / 2, NET_Y - SERVICE_FROM_NET), (COURT_W / 2, NET_Y + SERVICE_FROM_NET)],
        ]
    return COURT_MODEL_LINES


def _support(H: np.ndarray, mask: np.ndarray) -> float:
    """support x coverage:
    - support: fraction of the rendered court model that lands on mask-white
    - coverage: fraction of mask-white INSIDE the court quad explained by the model
    Coverage kills the self-similar half-court fit (its interior real lines — service and
    center lines — go unexplained)."""
    Hi = np.linalg.inv(H)
    h, w = mask.shape
    # thin centerline render of the model; support = model samples NEAR white (dilated mask)
    canvas = np.zeros_like(mask)
    thickness = _scaled_px(1, mask)
    for a, b in _court_model():
        p = cv2.perspectiveTransform(np.float32([[a, b]]), Hi)[0]
        cv2.line(canvas, tuple(np.int32(p[0])), tuple(np.int32(p[1])), 255, thickness)
    near = _scaled_px(5, mask, odd=True)
    kernel = np.ones((near, near), np.uint8)
    mask_d = cv2.dilate(mask, kernel)
    model_on = canvas > 0
    n_model = int(model_on.sum())
    scale = res.pixel_length(1, res.CANONICAL_SIZE, _image_size(mask))
    if n_model < 300 * scale:
        return 0.0
    support = float((mask_d[model_on] > 0).mean())
    # coverage: white inside the quad must be NEAR a (dilated) model line
    quad = cv2.perspectiveTransform(
        np.float32([[[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]]]), Hi
    )[0]
    qmask = np.zeros_like(mask)
    cv2.fillPoly(qmask, [np.int32(quad)], 255)
    coverage_near = _scaled_px(9, mask, odd=True)
    canvas_d = cv2.dilate(canvas, np.ones((coverage_near, coverage_near), np.uint8))
    inside = (qmask > 0) & (mask > 0)
    n_inside = int(inside.sum())
    if n_inside < 300 * scale * scale:
        return 0.0
    coverage = float((canvas_d[inside] > 0).mean())
    return support * coverage


SW = (COURT_W - 8.23) / 2  # singles sideline inset
# on-plane horizontal court lines a detected horizontal may correspond to (no net: the
# detected "net line" is the tape ~1m above the plane, unusable for the ground homography)
H_CANDS = [0.0, NET_Y - SERVICE_FROM_NET, NET_Y + SERVICE_FROM_NET, COURT_L]
V_CANDS = [(0.0, COURT_W), (SW, COURT_W - SW)]  # doubles pair, singles pair


def _standard_end_court_geometry(
    homography: np.ndarray,
    image_width: int,
    image_height: int,
) -> bool:
    geometry = assess_court_homography(homography, image_width, image_height)
    if not geometry["valid"]:
        return False
    centers = geometry["line_centers_y"]
    near = centers["near_baseline"]
    far = centers["far_baseline"]
    return (
        0.62 * image_height <= near <= 0.98 * image_height
        and far <= 0.36 * image_height
        and near - far >= 0.40 * image_height
    )


def _observed_net_band(img: np.ndarray) -> tuple[float, float, float] | None:
    """Return the dominant long central horizontal as ``(left, right, row)``.

    A broadcast net spans wider than the painted singles court and usually contributes
    several near-horizontal tape/mesh edges. This measurement is independent of a court
    hypothesis, preventing a bad homography from finding an unrelated line near its own
    projected net row.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    height, width = gray.shape
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(40, round(0.04 * width)),
        minLineLength=max(100, round(0.32 * width)),
        maxLineGap=max(20, round(0.07 * width)),
    )
    candidates = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4) if lines is not None else []:
        if x1 > x2:
            x1, x2, y1, y2 = x2, x1, y2, y1
        span = float(x2 - x1)
        row = 0.5 * float(y1 + y2)
        slope = abs(float(y2 - y1)) / max(span, 1.0)
        if 0.20 * height <= row <= 0.60 * height and slope <= 0.10:
            candidates.append((span, float(x1), float(x2), row))
    if not candidates:
        return None
    maximum_span = max(row[0] for row in candidates)
    dominant = [row for row in candidates if row[0] >= 0.90 * maximum_span]
    return (
        float(np.median([row[1] for row in dominant])),
        float(np.median([row[2] for row in dominant])),
        float(np.median([row[3] for row in dominant])),
    )


def _net_band_agrees(img: np.ndarray, homography: np.ndarray) -> bool:
    observed = _observed_net_band(img)
    if observed is None:
        return False
    left, right, tape_row = observed
    project = cv2.perspectiveTransform(
        np.float32([[[0.0, NET_Y], [COURT_W, NET_Y]]]),
        np.linalg.inv(homography),
    )[0]
    ground_row = float(np.mean(project[:, 1]))
    projected_left, projected_right = sorted(float(value) for value in project[:, 0])
    overlap = max(0.0, min(right, projected_right) - max(left, projected_left))
    projected_span = max(projected_right - projected_left, 1.0)
    height = img.shape[0]
    return (
        0.005 * height <= ground_row - tape_row <= 0.14 * height
        and overlap >= 0.70 * projected_span
    )


def find_court_h(
    img: np.ndarray,
    min_support: float = 0.25,
    top_fraction: float = 0.22,
    require_net_evidence: bool = False,
) -> np.ndarray:
    """Best ground homography by model support x coverage over line-to-model assignments.

    Detected ~horizontal lines may be either baseline or either service line; detected
    steep lines may be the doubles or singles sideline pair. Enumerate assignments, build
    H from the 4 intersections, keep the best-scoring."""
    mask = line_mask(img, top_fraction)
    h, w = mask.shape
    vote_threshold = _scaled_px(55, mask)
    max_gap = _scaled_px(18, mask)
    segs = cv2.HoughLinesP(
        mask, 1, np.pi / 180, threshold=vote_threshold, minLineLength=w // 14, maxLineGap=max_gap
    )
    if segs is None:
        raise SystemExit("no lines found")
    horiz, vert = [], []
    for s in segs.reshape(-1, 4):
        x1, y1, x2, y2 = (int(v) for v in s)
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        L = np.hypot(x2 - x1, y2 - y1)
        if ang < 20:
            horiz.append(((x1, y1, x2, y2), L))
        elif ang > 35:
            vert.append(((x1, y1, x2, y2), L))
    net_options = [
        (segment, length)
        for segment, length in horiz
        if 0.18 * h <= 0.5 * (segment[1] + segment[3]) <= 0.55 * h
    ]
    net_options.sort(key=lambda item: -item[1])
    net_y = 0.5 * (net_options[0][0][1] + net_options[0][0][3]) if net_options else 0.55 * h
    horiz = [s for s, _ in sorted(horiz, key=lambda t: -t[1])[:30]]
    vert = [s for s, _ in sorted(vert, key=lambda t: -t[1])[:16]]
    if len(horiz) < 2 or len(vert) < 2:
        raise SystemExit(f"line families too small: h={len(horiz)} v={len(vert)}")

    best_H, best_sc = None, 0.0
    ranked: list[tuple[float, int, np.ndarray]] = []
    candidate_id = 0
    for i in range(len(horiz)):
        for j in range(len(horiz)):
            ha, hb = horiz[i], horiz[j]
            ya, yb = (ha[1] + ha[3]) / 2, (hb[1] + hb[3]) / 2
            if ya > min(h * 0.36, net_y - h * 0.04) or yb < h * 0.62:
                continue
            for ca in range(len(H_CANDS) - 1):
                for cb in range(ca + 1, len(H_CANDS)):
                    # farther image line -> larger court y
                    ymap = {id(ha): H_CANDS[cb], id(hb): H_CANDS[ca]}
                    for u in range(len(vert)):
                        for v in range(len(vert)):
                            va, vb = vert[u], vert[v]
                            xa, xb = (va[0] + va[2]) / 2, (vb[0] + vb[2]) / 2
                            if xb - xa < w * 0.2:
                                continue
                            for xl, xr in V_CANDS:
                                src, dst = [], []
                                ok = True
                                for hline in (ha, hb):
                                    for vline, xc in ((va, xl), (vb, xr)):
                                        p = _intersect(hline, vline)
                                        if p is None or not (
                                            -w * 0.5 <= p[0] <= w * 1.5
                                            and -h * 0.5 <= p[1] <= h * 1.5
                                        ):
                                            ok = False
                                            break
                                        src.append(p)
                                        dst.append((xc, ymap[id(hline)]))
                                    if not ok:
                                        break
                                if not ok:
                                    continue
                                H = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
                                if not _standard_end_court_geometry(H, w, h):
                                    continue
                                sc = _support(H, mask)
                                if require_net_evidence and sc >= min(0.10, min_support):
                                    candidate_id += 1
                                    heapq.heappush(ranked, (sc, candidate_id, H))
                                    if len(ranked) > 64:
                                        heapq.heappop(ranked)
                                if sc > best_sc:
                                    best_H, best_sc = H, sc
    if best_H is None or best_sc < min_support:
        raise SystemExit(f"no assignment with sufficient support (best {best_sc:.2f})")
    if require_net_evidence:
        for _score, _candidate_id, candidate in sorted(ranked, reverse=True):
            if _net_band_agrees(img, candidate):
                return candidate
        raise SystemExit("no supported assignment agrees with the observed net band")
    return best_H


def find_court_h_native(
    img: np.ndarray,
    min_support: float = 0.25,
    *,
    require_net_evidence: bool = False,
) -> np.ndarray:
    """Estimate at the calibrated 960x540 scale, then map back to native pixels."""
    image_size = _image_size(img)
    canonical = (
        img
        if image_size == res.CANONICAL_SIZE
        else cv2.resize(
            img,
            (res.CANONICAL_SIZE.width, res.CANONICAL_SIZE.height),
            interpolation=cv2.INTER_AREA,
        )
    )
    failures = []
    for top_fraction in (0.08, 0.22):
        try:
            homography_at_canonical = find_court_h(
                canonical,
                min_support,
                top_fraction,
                require_net_evidence,
            )
            return res.image_to_world_homography(
                homography_at_canonical,
                res.CANONICAL_SIZE,
                image_size,
            )
        except SystemExit as error:
            failures.append(str(error))
    raise SystemExit("; ".join(failures))


def homography(corners: dict) -> np.ndarray:
    src = np.float32([corners["near_l"], corners["near_r"], corners["far_l"], corners["far_r"]])
    dst = np.float32([[0, 0], [COURT_W, 0], [0, COURT_L], [COURT_W, COURT_L]])
    return cv2.getPerspectiveTransform(src, dst)


def draw_validation(img: np.ndarray, H: np.ndarray, path: str) -> None:
    Hi = np.linalg.inv(H)

    def to_img(pts):
        p = cv2.perspectiveTransform(np.float32([pts]), Hi)[0]
        return [tuple(map(int, q)) for q in p]

    vis = img.copy()
    lines = [
        [(0, 0), (COURT_W, 0)],
        [(0, COURT_L), (COURT_W, COURT_L)],  # baselines
        [(0, 0), (0, COURT_L)],
        [(COURT_W, 0), (COURT_W, COURT_L)],  # doubles side
        [(0, NET_Y), (COURT_W, NET_Y)],  # net
        [(0, NET_Y - SERVICE_FROM_NET), (COURT_W, NET_Y - SERVICE_FROM_NET)],  # near svc
        [(0, NET_Y + SERVICE_FROM_NET), (COURT_W, NET_Y + SERVICE_FROM_NET)],  # far svc
        [(COURT_W / 2, NET_Y - SERVICE_FROM_NET), (COURT_W / 2, NET_Y + SERVICE_FROM_NET)],
    ]
    for a, b in lines:
        pa, pb = to_img([a, b])
        cv2.line(vis, pa, pb, (0, 0, 255), 2)
    cv2.imwrite(path, vis)


def per_point_homographies(
    out_dir: str,
    frames_dir: str = "rally_frames",
    *,
    allow_neighbor_fallback: bool = False,
) -> None:
    """Estimate H per point clip without hidden assistance by default."""
    clips = sorted(glob.glob(os.path.join(out_dir, frames_dir, "pt*")))
    results, n_ok = {}, 0
    for clip in clips:
        frames = sorted(glob.glob(os.path.join(clip, "f_*.jpg")))
        if not frames:
            continue
        pt = int(os.path.basename(clip)[2:])
        H = None
        frame_indices = [
            round((len(frames) - 1) * quantile) for quantile in (0.3, 0.7, 0.5, 0.25, 0.9)
        ]
        for fp in (frames[index] for index in dict.fromkeys(frame_indices)):
            img = cv2.imread(fp)
            if img is None:
                continue
            try:
                candidate = find_court_h_native(img)
                geometry = assess_court_homography(
                    candidate,
                    img.shape[1],
                    img.shape[0],
                )
                if geometry["valid"]:
                    H = candidate
                    break
            except SystemExit:
                continue
        if H is not None:
            results[pt] = H
            n_ok += 1
    print(f"homography ok on {n_ok}/{len(clips)} clips")
    pts_all = [int(os.path.basename(c)[2:]) for c in clips]
    filled = {}
    ok_pts = sorted(results)
    for pt in pts_all:
        if pt in results:
            filled[pt] = results[pt]
        elif allow_neighbor_fallback and ok_pts:
            nearest = min(ok_pts, key=lambda q: abs(q - pt))
            filled[pt] = results[nearest]
        else:
            filled[pt] = np.full((3, 3), np.nan, dtype=float)
    np.savez_compressed(
        os.path.join(out_dir, "court_H_per_point.npz"),
        pts=np.array(sorted(filled)),
        H=np.stack([filled[p] for p in sorted(filled)]),
    )
    # validation overlay on a middle clip
    if ok_pts:
        mid = ok_pts[len(ok_pts) // 2]
        clip = os.path.join(out_dir, frames_dir, f"pt{mid:04d}")
        fp = sorted(glob.glob(os.path.join(clip, "f_*.jpg")))
        img = cv2.imread(fp[len(fp) // 2])
        draw_validation(img, results[mid], os.path.join(out_dir, "court_validation.jpg"))
        print(f"validation overlay (pt {mid}) -> court_validation.jpg")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames-dir", default="rally_frames")
    ap.add_argument("--allow-neighbor-fallback", action="store_true")
    args = ap.parse_args()
    per_point_homographies(
        args.out,
        args.frames_dir,
        allow_neighbor_fallback=args.allow_neighbor_fallback,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
