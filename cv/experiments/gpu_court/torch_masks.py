"""Court-line masks of ``court_topology.court_line_masks`` for a batch of frames on the GPU.

Question: can the two painted-line proposal masks (``court.line_mask`` and
``court_topology.relaxed_chroma_line_mask``) be computed for a whole clip at once on the GPU
with the same bytes as OpenCV?

Everything here is integer arithmetic that mirrors OpenCV's 8-bit code paths:

- BGR2GRAY and HSV saturation are per-pixel functions of (B, G, R). They are looked up in
  tables that the installed cv2 builds over all 2^24 colours at first use. The fixed-point
  formula from the OpenCV source misses 43,864 colours on this build, so the tables are the
  exact route;
- V is ``max(B, G, R)``;
- MORPH_TOPHAT is ``src - dilate(erode(src))`` with a rectangular kernel. The border is ignored,
  matching the default constant border of +inf for erode and -inf for dilate.

``test_torch_masks.py`` checks byte equality against cv2 on real frames.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from cv.pipeline import court

_LUTS: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def _luts(device) -> tuple[torch.Tensor, torch.Tensor]:
    """Gray and HSV-saturation tables over every (B, G, R), computed by cv2 itself."""
    key = str(device)
    if key not in _LUTS:
        import cv2

        index = np.arange(1 << 24, dtype=np.uint32)
        colours = np.stack([(index >> 16) & 255, (index >> 8) & 255, index & 255], -1)
        image = colours.astype(np.uint8).reshape(4096, 4096, 3)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape(-1)
        saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 1].reshape(-1)
        _LUTS[key] = (torch.from_numpy(gray).to(device), torch.from_numpy(saturation).to(device))
    return _LUTS[key]


def gray_sv(bgr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(B, H, W, 3) uint8 -> gray, saturation, value as int32 (cv2.cvtColor 8-bit)."""
    b, g, r = (bgr[..., k].to(torch.int32) for k in range(3))
    index = ((b << 16) | (g << 8) | r).long()
    gray_table, saturation_table = _luts(bgr.device)
    gray = gray_table[index].to(torch.int32)
    s = saturation_table[index].to(torch.int32)
    v = torch.maximum(torch.maximum(b, g), r)
    return gray, s, v


def tophat(gray: torch.Tensor, kernel: int) -> torch.Tensor:
    """MORPH_TOPHAT with a kernel x kernel rectangle (anchor centre), border ignored.

    A rectangle's min/max filter is separable exactly: a row pass, then a column pass.
    """
    pad = kernel // 2
    x = gray.to(torch.float32)[:, None]

    def max_filter(t):
        t = F.max_pool2d(F.pad(t, (pad, pad, 0, 0), value=-1e9), (1, kernel), stride=1)
        return F.max_pool2d(F.pad(t, (0, 0, pad, pad), value=-1e9), (kernel, 1), stride=1)

    eroded = -max_filter(-x)
    opened = max_filter(eroded)
    return (x - opened)[:, 0].to(torch.int32)


def _kernels(width: int, height: int) -> tuple[int, int]:
    probe = np.empty((height, width, 3), np.uint8)
    standard = court._scaled_px(15, probe, odd=True)
    relaxed = max(9, int(round(15 * width / 960)))
    if relaxed % 2 == 0:
        relaxed += 1
    return standard, relaxed


@torch.no_grad()
def court_line_masks_batch(bgr: torch.Tensor, top_fraction: float = 0.12) -> torch.Tensor:
    """(B, H, W, 3) uint8 -> (B, 2, H, W) uint8 masks (0/255): standard, relaxed_chroma."""
    B, H, W, _ = bgr.shape
    gray, s, v = gray_sv(bgr)
    k_standard, k_relaxed = _kernels(W, H)
    observed = torch.from_numpy(court.line_mask_observation_mask((H, W), top_fraction)).to(
        bgr.device
    )
    standard = (tophat(gray, k_standard) > 38) & (s < 90) & observed
    relaxed = (tophat(gray, k_relaxed) > 22) & (s < 190) & (v > 105) & observed
    return torch.stack([standard, relaxed], 1).to(torch.uint8) * 255


class DeviceMask:
    """A GPU-resident uint8 mask. ``.shape`` is free; anything that reads the pixels
    (``np.asarray(mask)``, indexing, ``==``) gets the host copy, made once on first use."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        self.shape = tuple(tensor.shape)
        self.dtype = np.dtype(np.uint8)
        self._host = None

    def host(self) -> np.ndarray:
        if self._host is None:
            self._host = self.tensor.cpu().numpy()
        return self._host

    def __array__(self, dtype=None, copy=None):
        array = self.host()
        return array if dtype is None else array.astype(dtype)

    def __getitem__(self, index):
        return self.host()[index]

    def __eq__(self, other):
        return self.host() == other


class DeviceDistance:
    """A GPU-resident distance image that ``topology_score`` reads like a NumPy array.

    It supports ``.shape`` and ``distance[rows, cols]``. The scorer only gathers a few hundred
    projected pixels per call, so this avoids copying 8 MB per mask back to the host. The values
    are the same float32 numbers the full array would hold.
    """

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        self.shape = tuple(tensor.shape)

    def __getitem__(self, index):
        rows, cols = index
        device = self.tensor.device
        rows = torch.as_tensor(np.asarray(rows), device=device)
        cols = torch.as_tensor(np.asarray(cols), device=device)
        return self.tensor[rows, cols].cpu().numpy()


@torch.no_grad()
def frame_support(images: list[np.ndarray], device=None) -> list[tuple]:
    """``court_line_masks`` and their ``_distance_samples`` for a chunk of same-size frames.

    Returns, per frame, ``(masks, distances)``:

    - ``masks`` as ``court_line_masks``, byte-identical, GPU-resident ``DeviceMask`` views (the
      scorers read only their shape once distances are given);
    - ``distances`` as GPU-resident ``DeviceDistance`` views of
      ``distanceTransform(mask == 0, DIST_L2, 3)`` from ``dt_cuda`` (one launch, same as
      ``torch_distance``).

    The distances are OpenCV's integer chamfer, within ~5e-4 px of this cv2 build's float path
    after the 100 px clip, so scores are compared on decisions and rounded values, not bytes.
    """
    from cv.experiments.gpu_court import dt_cuda

    if not images:
        return []
    device = device or torch.device("cuda")
    batch = torch.empty((len(images), *images[0].shape), dtype=torch.uint8, device=device)
    for k, image in enumerate(images):  # no host-side stack copy
        batch[k].copy_(torch.from_numpy(image))
    masks = court_line_masks_batch(batch)
    B, _, H, W = masks.shape
    distances = dt_cuda.distance_transform_tensor((masks == 0).reshape(B * 2, H, W))
    distances = distances.reshape(B, 2, H, W)
    return [
        (
            (("standard", DeviceMask(masks[k, 0])), ("relaxed_chroma", DeviceMask(masks[k, 1]))),
            (DeviceDistance(distances[k, 0]), DeviceDistance(distances[k, 1])),
        )
        for k in range(B)
    ]
