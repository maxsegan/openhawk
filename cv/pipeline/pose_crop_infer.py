"""Shared tracked-player crop pose inference.

Protocol: a square context window around the (seeded) player is pasted at an
explicit presentation scale onto a fixed gray canvas (default 640, the pose
model's native training resolution) so the model NEVER resamples the input
implicitly — scale is a controlled experimental variable. Variants
(scale x hflip) are batched together; per-keypoint results are merged by
confidence-weighted averaging in native window coordinates.

Keypoint order follows pose.KEYPOINT_NAMES (17 COCO keypoints).
"""

from __future__ import annotations

import numpy as np
import cv2

CANVAS = 640
GRAY = 114
# COCO left/right pairs among the 17 keypoints (for hflip un-mirroring)
COCO_FLIP = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))


def window_for_box(box_xyxy, img_w: int, img_h: int, factor: float = 3.0) -> tuple[int, int, int]:
    """Square window (x0, y0, side) around the box center, clamped to the image."""
    x0, y0, x1, y1 = box_xyxy
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = int(round(max(y1 - y0, (x1 - x0) * 0.75) * factor))
    side = int(min(max(side, 96), min(img_w, img_h)))
    wx = int(round(cx - side / 2))
    wy = int(round(cy - side / 2))
    wx = max(0, min(wx, img_w - side))
    wy = max(0, min(wy, img_h - side))
    return wx, wy, side


def make_variant(
    window: np.ndarray,
    scale: float,
    flip: bool,
    canvas: int = CANVAS,
    focus_xy: tuple[float, float] | None = None,
):
    """Render one canvas variant. Returns (image, meta) where meta maps canvas->window."""
    side = window.shape[0]
    target = int(round(side * scale))
    if target > canvas:
        keep = int(np.floor(canvas / scale))
        if focus_xy is None:
            crop_x = crop_y = (side - keep) // 2
        else:
            crop_x = int(round(focus_xy[0] - keep / 2))
            crop_y = int(round(focus_xy[1] - keep / 2))
            crop_x = max(0, min(crop_x, side - keep))
            crop_y = max(0, min(crop_y, side - keep))
        window = window[crop_y : crop_y + keep, crop_x : crop_x + keep]
        crop_off = (crop_x, crop_y)
        target = int(round(keep * scale))
    else:
        crop_off = (0, 0)
    scaled = (
        cv2.resize(window, (target, target), interpolation=cv2.INTER_CUBIC)
        if scale != 1.0
        else window
    )
    if flip:
        scaled = scaled[:, ::-1]
    image = np.full((canvas, canvas, 3), GRAY, dtype=np.uint8)
    px = (canvas - target) // 2
    py = (canvas - target) // 2
    image[py : py + target, px : px + target] = scaled
    meta = {
        "scale": scale,
        "flip": flip,
        "paste_xy": (px, py),
        "paste_side": target,
        "crop_off": crop_off,
    }
    return image, meta


def canvas_to_window(xy: np.ndarray, meta) -> np.ndarray:
    """Map (M,2) canvas coords back to window coords (un-flip, un-scale)."""
    out = xy.astype(np.float64).copy()
    px, py = meta["paste_xy"]
    if meta["flip"]:
        out[:, 0] = (px + meta["paste_side"]) - (out[:, 0] - px) - 1
    crop_x, crop_y = meta["crop_off"]
    out[:, 0] = (out[:, 0] - px) / meta["scale"] + crop_x
    out[:, 1] = (out[:, 1] - py) / meta["scale"] + crop_y
    return out


def focus_box_to_canvas(box, meta) -> np.ndarray:
    """Map an observed window-pixel box through the exact presentation transform."""
    box = np.asarray(box, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError("finite nonempty source focus box in window pixels required")
    corners = box.reshape(2, 2).copy()
    corners -= np.asarray(meta["crop_off"])
    corners *= meta["scale"]
    corners += np.asarray(meta["paste_xy"])
    if meta["flip"]:
        px = meta["paste_xy"][0]
        corners[:, 0] = px + meta["paste_side"] - (corners[:, 0] - px) - 1
    return np.concatenate([corners.min(axis=0), corners.max(axis=0)])


def select_detection(result, canvas: int = CANVAS, focus_box=None):
    """Bind a pose to its supplied source box; retain the old unseeded fallback.

    With a focus box, highest image IoU owns the participant. Confidence breaks
    exact overlap ties only. Disjoint detections never substitute another person.
    """
    if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        return None
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    if focus_box is not None:
        focus = np.asarray(focus_box, dtype=float)
        if focus.shape != (4,) or not np.isfinite(focus).all() or np.any(focus[2:] <= focus[:2]):
            raise ValueError("finite nonempty focus box in canvas pixels required")
        intersection = np.maximum(
            0.0, np.minimum(boxes[:, 2:], focus[2:]) - np.maximum(boxes[:, :2], focus[:2])
        ).prod(axis=1)
        area = np.maximum(0.0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
        union = area + np.prod(focus[2:] - focus[:2]) - intersection
        overlap = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        keep = np.isfinite(boxes).all(axis=1) & np.isfinite(confs) & (overlap > 0)
        if not keep.any():
            return None
        candidates = np.flatnonzero(keep)
        best = int(candidates[np.lexsort((confs[candidates], overlap[candidates]))[-1]])
    else:
        centers = np.stack(
            [(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1
        )
        dist = np.linalg.norm(centers - canvas / 2, axis=1) / (canvas / 2)
        score = confs - 0.5 * dist
        keep = dist < 0.85
        if not keep.any():
            return None
        score[~keep] = -np.inf
        best = int(score.argmax())
    kpts = result.keypoints.data[best].detach().cpu().numpy()  # (17,3)
    return {"box": boxes[best], "conf": float(confs[best]), "kpts": kpts}


def unflip_keypoints(kpts: np.ndarray) -> np.ndarray:
    """Swap left/right keypoint identities after horizontal un-mirroring."""
    out = kpts.copy()
    for a, b in COCO_FLIP:
        out[[a, b]] = out[[b, a]]
    return out


def merge_variants(detections: list[tuple[dict, dict]], min_det_conf: float = 0.15):
    """Merge per-variant detections (already mapped meta) into one window-coord pose.

    detections: list of (selected_detection, variant_meta). Returns dict with
    box (window coords), det_conf, kpts (17,3) or None if nothing usable.
    """
    usable = [(det, meta) for det, meta in detections if det and det["conf"] >= min_det_conf]
    if not usable:
        return None
    kpt_stack, kw_stack, boxes, det_confs = [], [], [], []
    for det, meta in usable:
        xy = canvas_to_window(det["kpts"][:, :2], meta)
        kconf = det["kpts"][:, 2].copy()
        if meta["flip"]:
            xy = unflip_keypoints(xy)
            kconf = unflip_keypoints(kconf[:, None])[:, 0]
        corners = canvas_to_window(
            np.array([[det["box"][0], det["box"][1]], [det["box"][2], det["box"][3]]]),
            meta,
        )
        boxes.append(
            [
                min(corners[0, 0], corners[1, 0]),
                min(corners[0, 1], corners[1, 1]),
                max(corners[0, 0], corners[1, 0]),
                max(corners[0, 1], corners[1, 1]),
            ]
        )
        det_confs.append(det["conf"])
        kpt_stack.append(xy)
        kw_stack.append(kconf * det["conf"])
    kpt_stack = np.stack(kpt_stack)  # (V,17,2)
    kw = np.stack(kw_stack)  # (V,17)
    weights = kw / np.maximum(kw.sum(axis=0, keepdims=True), 1e-9)
    merged_xy = (kpt_stack * weights[:, :, None]).sum(axis=0)
    det_confs = np.asarray(det_confs)
    merged_kconf = (
        np.stack(
            [
                d[0]["kpts"][:, 2]
                if not d[1]["flip"]
                else unflip_keypoints(d[0]["kpts"][:, 2:3])[:, 0]
                for d in usable
            ]
        )
        * det_confs[:, None]
    ).sum(axis=0) / det_confs.sum()
    box_w = det_confs / det_confs.sum()
    merged_box = (np.asarray(boxes) * box_w[:, None]).sum(axis=0)
    return {
        "box": merged_box,
        "det_conf": float(det_confs.max()),
        "kpts_xy": merged_xy,
        "kpts_conf": merged_kconf,
        "n_variants": len(usable),
    }


def pose_windows(
    model,
    windows: list[np.ndarray],
    scales=(1.0,),
    flips=(False,),
    device: str = "0",
    batch: int = 96,
    canvas: int = CANVAS,
    conf: float = 0.05,
    focus_boxes: list[list[float]] | None = None,
    half: bool = False,
):
    """Run the variant grid over many windows. Returns one merged result per window."""
    variant_images, variant_meta, owner, variant_focus = [], [], [], []
    for wi, window in enumerate(windows):
        focus = None
        if focus_boxes is not None:
            x0, y0, x1, y1 = focus_boxes[wi]
            focus = ((x0 + x1) / 2, (y0 + y1) / 2)
        for scale in scales:
            for flip in flips:
                image, meta = make_variant(window, scale, flip, canvas, focus_xy=focus)
                variant_images.append(image)
                variant_meta.append(meta)
                owner.append(wi)
                variant_focus.append(
                    None if focus_boxes is None else focus_box_to_canvas(focus_boxes[wi], meta)
                )
    selections: list = [None] * len(variant_images)
    for start in range(0, len(variant_images), batch):
        chunk = variant_images[start : start + batch]
        inference_options = {
            "imgsz": canvas,
            "conf": conf,
            "device": device,
            "verbose": False,
        }
        if half:
            inference_options["half"] = True
        results = model.predict(chunk, **inference_options)
        for offset, result in enumerate(results):
            selections[start + offset] = select_detection(
                result, canvas, variant_focus[start + offset]
            )
    merged: list = [None] * len(windows)
    per_window: dict[int, list] = {}
    for sel, meta, wi in zip(selections, variant_meta, owner):
        per_window.setdefault(wi, []).append((sel, meta))
    for wi, dets in per_window.items():
        merged[wi] = merge_variants(dets)
    return merged
