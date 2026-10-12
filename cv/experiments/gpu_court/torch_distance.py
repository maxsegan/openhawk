"""Bit-exact ``cv2.distanceTransform(src, cv2.DIST_L2, 3)`` for a batch of images on the GPU.

OpenCV's 3x3 L2 transform is integer chamfer arithmetic (``distransform.cpp``):

- weights HV = round(0.955 * 2^16) and DIAG = round(1.3693 * 2^16);
- a forward raster pass (top-left, top, top-right, left);
- a backward pass (bottom-right, bottom, bottom-left, right);
- output ``float(d) * 2^-16``.

Each pass is row-sequential, but within a row the left (right) dependency is a min-plus prefix
(suffix) scan, ``cur[j] = min_k (v[k] + HV * |j - k|)``. That scan is exact in integers. So every
row is a handful of vector operations over the whole batch, and the result equals OpenCV's
bit for bit (``test_torch_distance.py``).

    from cv.experiments.gpu_court.torch_distance import distance_transform
    distances = distance_transform(sources)  # (B, H, W) uint8/bool, nonzero = background
"""

from __future__ import annotations

import numpy as np
import torch

SHIFT = 16
HV = int(np.rint(np.float32(0.955) * np.float32(1 << SHIFT)))
DIAG = int(np.rint(np.float32(1.3693) * np.float32(1 << SHIFT)))
DIST_MAX = 0xFFFFFFFF - DIAG
SCALE = np.float32(1.0 / (1 << SHIFT))


@torch.no_grad()
def distance_transform_tensor(src: torch.Tensor) -> torch.Tensor:
    """(B, H, W) nonzero-background sources -> (B, H, W) float32 distances (OpenCV DIST_L2, 3)."""
    B, H, W = src.shape
    device = src.device
    background = src != 0
    ramp = torch.arange(W, device=device, dtype=torch.int64) * HV
    forward = torch.empty((B, H, W), dtype=torch.int64, device=device)
    prev = torch.full((B, W + 2), DIST_MAX, dtype=torch.int64, device=device)
    for i in range(H):
        v = torch.minimum(
            torch.minimum(prev[:, :-2] + DIAG, prev[:, 1:-1] + HV), prev[:, 2:] + DIAG
        )
        v = torch.where(background[:, i], v, torch.zeros_like(v))
        cur = torch.cummin(v - ramp, dim=1).values + ramp
        cur = torch.clamp(cur, max=DIST_MAX)
        forward[:, i] = cur
        prev[:, 1:-1] = cur
    out = torch.empty((B, H, W), dtype=torch.float32, device=device)
    below = torch.full((B, W + 2), DIST_MAX, dtype=torch.int64, device=device)
    for i in range(H - 1, -1, -1):
        w = torch.minimum(forward[:, i], below[:, 2:] + DIAG)
        w = torch.minimum(w, below[:, 1:-1] + HV)
        w = torch.minimum(w, below[:, :-2] + DIAG)
        # right scan: cur[j] = min(w[j], cur[j+1] + HV)
        cur = torch.flip(torch.cummin(torch.flip(w + ramp, [1]), dim=1).values, [1]) - ramp
        below[:, 1:-1] = cur
        out[:, i] = cur.to(torch.float32) * float(SCALE)
    return out


def distance_transform(sources: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    """NumPy (B, H, W) -> float32 (B, H, W), as ``cv2.distanceTransform(s, DIST_L2, 3)`` each."""
    device = device or torch.device("cuda")
    src = torch.from_numpy(np.ascontiguousarray(sources)).to(device)
    return distance_transform_tensor(src).cpu().numpy()
