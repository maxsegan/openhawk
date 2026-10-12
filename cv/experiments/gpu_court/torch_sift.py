"""OpenCV-compatible SIFT on the GPU (PyTorch), for court registration.

Question: can ``cv2.SIFT_create(nfeatures=4000).detectAndCompute(gray, mask)`` (0.3-0.6 s per
1080p frame on one core, ~63% of court_geometry CPU) run on the GPU with the same features?

This mirrors OpenCV 5's float SIFT step by step (``sift.dispatch.cpp`` / ``sift.simd.hpp``):
doubled first octave (bilinear), separable Gaussian blurs with OpenCV's kernel size and
BORDER_REFLECT_101, nearest-neighbour octave halving, DoG, 26-neighbour extrema above the
integer threshold, up to five quadratic refinements with contrast and edge tests, 36-bin
smoothed orientation histograms with 0.8 peak ratio, duplicate removal, ``retainBest`` with
ties, pixel-mask filtering, and the 4x4x8 trilinear descriptor with 0.2 clipping and the
512 scale saturated to 0..255. Everything is float32 with TF32 off.

It is not bit-identical: GPU float summation order, ``atan2``/``exp`` versus OpenCV's fast
approximations, and LU rounding move a few keypoints across thresholds. ``parity.py`` measures
the agreement, and the court pipeline judges the result by homographies and camera outputs.

    from cv.experiments.gpu_court.torch_sift import detect_and_compute
    keypoints, descriptors = detect_and_compute(gray_uint8, mask_or_None, nfeatures=4000)
"""

from __future__ import annotations

import math
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

N_LAYERS = 3
CONTRAST = 0.04
EDGE = 10.0
SIGMA = 1.6
INIT_SIGMA = 0.5
BORDER = 5
MAX_STEPS = 5
ORI_BINS = 36
ORI_SIG = 1.5
ORI_RADIUS = 4.5
ORI_PEAK = 0.8
D, NB = 4, 8
DESCR_SCL = 3.0
DESCR_MAG = 0.2
INT_FCTR = 512.0
FLT_EPS = float(np.finfo(np.float32).eps)

_KERNELS: dict[tuple, torch.Tensor] = {}


def _device() -> torch.device:
    """``TENNIS_COURT_SIFT_DEVICE`` (index among visible CUDA devices, default 0)."""
    return torch.device("cuda", int(os.environ.get("TENNIS_COURT_SIFT_DEVICE", "0")))


def _gaussian_kernel(sigma: float, device) -> torch.Tensor:
    """cv2.getGaussianKernel for a float image (ksize = round(sigma*8+1) | 1)."""
    size = int(round(sigma * 8 + 1)) | 1
    key = (size, sigma, str(device))
    if key not in _KERNELS:
        kernel = cv2.getGaussianKernel(size, sigma, cv2.CV_32F).ravel()
        _KERNELS[key] = torch.from_numpy(kernel.copy()).to(device)
    return _KERNELS[key]


def _blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of an (H, W) float32 image, BORDER_REFLECT_101."""
    kernel = _gaussian_kernel(sigma, image.device)
    radius = kernel.numel() // 2
    x = image[None, None]
    h, w = image.shape
    # reflect101 needs pad < size; tiny octaves fall back to replicate-safe slicing below.
    pad_w = min(radius, w - 1)
    pad_h = min(radius, h - 1)
    if pad_w < radius or pad_h < radius:
        x = _reflect101_large(x, radius)
        x = F.conv2d(x, kernel.view(1, 1, 1, -1))
        x = F.conv2d(x, kernel.view(1, 1, -1, 1))
        return x[0, 0]
    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, kernel.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(x, kernel.view(1, 1, -1, 1))
    return x[0, 0]


def _reflect101_large(x: torch.Tensor, radius: int) -> torch.Tensor:
    """BORDER_REFLECT_101 padding by index arithmetic (any pad width)."""
    h, w = x.shape[-2:]

    def index(n):
        i = torch.arange(-radius, n + radius, device=x.device)
        if n == 1:
            return torch.zeros_like(i)
        period = 2 * (n - 1)
        i = torch.remainder(i, period)
        return torch.where(i >= n, period - i, i)

    return x[..., index(h)[:, None], index(w)[None, :]]


def _pyramid(gray: torch.Tensor):
    """Gaussian and DoG pyramids of the doubled image; lists per octave of (L, H, W)."""
    base = gray.float()
    up = F.interpolate(base[None, None], scale_factor=2, mode="bilinear", align_corners=False)[0, 0]
    sig_diff = math.sqrt(max(SIGMA * SIGMA - INIT_SIGMA * INIT_SIGMA * 4, 0.01))
    first = _blur(up, float(np.float32(sig_diff)))
    n_octaves = int(round(math.log(min(first.shape)) / math.log(2.0) - 2)) + 1
    sig = [SIGMA]
    k = 2 ** (1.0 / N_LAYERS)
    for i in range(1, N_LAYERS + 3):
        prev = k ** (i - 1) * SIGMA
        total = prev * k
        sig.append(math.sqrt(total * total - prev * prev))
    gauss, dog = [], []
    for o in range(n_octaves):
        layers = []
        for i in range(N_LAYERS + 3):
            if o == 0 and i == 0:
                layers.append(first)
            elif i == 0:
                src = gauss[o - 1][N_LAYERS]
                h, w = src.shape
                layers.append(src[0 : 2 * (h // 2) : 2, 0 : 2 * (w // 2) : 2].contiguous())
            else:
                layers.append(_blur(layers[-1], sig[i]))
        stack = torch.stack(layers)
        gauss.append(stack)
        dog.append(stack[1:] - stack[:-1])
    return gauss, dog


def _extrema(dog: torch.Tensor, threshold: float) -> torch.Tensor:
    """(layer, r, c) of 26-neighbour extrema (non-strict, as OpenCV) in layers 1..N_LAYERS."""
    x = dog[None, None]
    mx = F.max_pool3d(x, 3, stride=1, padding=1)[0, 0]
    mn = -F.max_pool3d(-x, 3, stride=1, padding=1)[0, 0]
    L, H, W = dog.shape
    valid = torch.zeros_like(dog, dtype=torch.bool)
    if H <= 2 * BORDER or W <= 2 * BORDER:
        return torch.empty((0, 3), dtype=torch.long, device=dog.device)
    valid[1 : N_LAYERS + 1, BORDER : H - BORDER, BORDER : W - BORDER] = True
    v = dog
    hit = valid & (v.abs() > threshold) & (((v > 0) & (v >= mx)) | ((v < 0) & (v <= mn)))
    return torch.nonzero(hit)


class _Packed:
    """All octaves' layers of one pyramid in one flat buffer, for batched gathers."""

    def __init__(self, stacks: list[torch.Tensor]):
        self.flat = torch.cat([s.reshape(-1) for s in stacks])
        sizes = [s.numel() for s in stacks]
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        device = stacks[0].device
        self.start = torch.tensor(starts, dtype=torch.long, device=device)
        self.H = torch.tensor([s.shape[1] for s in stacks], dtype=torch.long, device=device)
        self.W = torch.tensor([s.shape[2] for s in stacks], dtype=torch.long, device=device)

    def at(self, octave, layer, r, c):
        W, H = self.W[octave], self.H[octave]
        return self.flat[self.start[octave] + (layer * H + r) * W + c]


def _refine(dogs: _Packed, octave: torch.Tensor, cand: torch.Tensor) -> dict:
    """adjustLocalExtrema for the candidates of all octaves at once; surviving rows."""
    H, W = dogs.H[octave], dogs.W[octave]
    img_scale = 1.0 / 255.0
    deriv, second, cross = img_scale * 0.5, img_scale, img_scale * 0.25
    layer, r, c = cand[:, 0].clone(), cand[:, 1].clone(), cand[:, 2].clone()
    n = len(cand)
    device = cand.device
    alive = torch.ones(n, dtype=torch.bool, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    xi = torch.zeros(n, device=device)
    xr = torch.zeros(n, device=device)
    xc = torch.zeros(n, device=device)
    step = torch.arange(-1, 2, device=device)

    def neighbourhood():
        ll = layer[:, None, None, None] + step[None, :, None, None]
        rr = r[:, None, None, None] + step[None, None, :, None]
        cc = c[:, None, None, None] + step[None, None, None, :]
        return dogs.at(octave[:, None, None, None], ll, rr, cc)

    def derivatives(v):
        def at(dl, dr, dc):
            return v[:, 1 + dl, 1 + dr, 1 + dc]

        dD = torch.stack(
            [
                (at(0, 0, 1) - at(0, 0, -1)) * deriv,
                (at(0, 1, 0) - at(0, -1, 0)) * deriv,
                (at(1, 0, 0) - at(-1, 0, 0)) * deriv,
            ],
            1,
        )
        v2 = at(0, 0, 0) * 2
        dxx = (at(0, 0, 1) + at(0, 0, -1) - v2) * second
        dyy = (at(0, 1, 0) + at(0, -1, 0) - v2) * second
        dss = (at(1, 0, 0) + at(-1, 0, 0) - v2) * second
        dxy = (at(0, 1, 1) - at(0, 1, -1) - at(0, -1, 1) + at(0, -1, -1)) * cross
        dxs = (at(1, 0, 1) - at(1, 0, -1) - at(-1, 0, 1) + at(-1, 0, -1)) * cross
        dys = (at(1, 1, 0) - at(1, -1, 0) - at(-1, 1, 0) + at(-1, -1, 0)) * cross
        return dD, dxx, dyy, dss, dxy, dxs, dys, at(0, 0, 0)

    for _ in range(MAX_STEPS):
        todo = alive & ~done
        dD, dxx, dyy, dss, dxy, dxs, dys, _ = derivatives(neighbourhood())
        hessian = torch.stack(
            [
                torch.stack([dxx, dxy, dxs], 1),
                torch.stack([dxy, dyy, dys], 1),
                torch.stack([dxs, dys, dss], 1),
            ],
            1,
        )
        X, info = torch.linalg.solve_ex(hessian, dD[..., None])
        X = torch.where((info == 0)[:, None, None], X, torch.zeros_like(X))[..., 0]
        xi = torch.where(todo, -X[:, 2], xi)
        xr = torch.where(todo, -X[:, 1], xr)
        xc = torch.where(todo, -X[:, 0], xc)
        converged = todo & (xi.abs() < 0.5) & (xr.abs() < 0.5) & (xc.abs() < 0.5)
        done |= converged
        moving = todo & ~converged
        huge = moving & ((xi.abs() > 7e8) | (xr.abs() > 7e8) | (xc.abs() > 7e8))
        alive &= ~huge
        moving &= ~huge
        c = torch.where(moving, c + torch.round(xc).long(), c)
        r = torch.where(moving, r + torch.round(xr).long(), r)
        layer = torch.where(moving, layer + torch.round(xi).long(), layer)
        out = moving & (
            (layer < 1)
            | (layer > N_LAYERS)
            | (c < BORDER)
            | (c >= W - BORDER)
            | (r < BORDER)
            | (r >= H - BORDER)
        )
        alive &= ~out
        layer = torch.where(alive, layer, torch.ones_like(layer))
        r = torch.where(alive, r, torch.full_like(r, BORDER))
        c = torch.where(alive, c, torch.full_like(c, BORDER))
    alive &= done
    dD, dxx, dyy, _, dxy, _, _, centre = derivatives(neighbourhood())
    t = dD[:, 0] * xc + dD[:, 1] * xr + dD[:, 2] * xi
    contr = centre * img_scale + t * 0.5
    alive &= ~(contr.abs() * N_LAYERS < CONTRAST)
    tr = dxx + dyy
    det = dxx * dyy - dxy * dxy
    alive &= ~((det <= 0) | (tr * tr * EDGE >= (EDGE + 1) * (EDGE + 1) * det))
    k = alive
    scale = (1 << octave[k]).float()
    return dict(
        octave=octave[k],
        layer=layer[k],
        r=r[k],
        c=c[k],
        x=(c[k].float() + xc[k]) * scale,
        y=(r[k].float() + xr[k]) * scale,
        xi=xi[k],
        size=SIGMA * torch.pow(2.0, (layer[k].float() + xi[k]) / N_LAYERS) * scale * 2,
        response=contr[k].abs(),
    )


def _window(gauss: _Packed, octave, layer, cy, cx, radius: int):
    """Gradients (dx, dy) and in-image flags on a (2R+1)^2 window per keypoint."""
    offs = torch.arange(-radius, radius + 1, device=cy.device)
    yy = cy[:, None, None] + offs[None, :, None]
    xx = cx[:, None, None] + offs[None, None, :]
    o = octave[:, None, None]
    H, W = gauss.H[o], gauss.W[o]
    inside = (yy > 0) & (yy < H - 1) & (xx > 0) & (xx < W - 1)
    y = torch.minimum(torch.clamp(yy, min=1), H - 2)
    x = torch.minimum(torch.clamp(xx, min=1), W - 2)
    lay = layer[:, None, None]
    dx = gauss.at(o, lay, y, x + 1) - gauss.at(o, lay, y, x - 1)
    dy = gauss.at(o, lay, y - 1, x) - gauss.at(o, lay, y + 1, x)
    return dx, dy, inside, offs


def _atan2_degrees(dy, dx):
    angle = torch.rad2deg(torch.atan2(dy, dx))
    return torch.where(angle < 0, angle + 360.0, angle)


def _orientations(gauss: _Packed, kp: dict) -> dict:
    """calcOrientationHist + peak selection for all keypoints; one row per accepted peak."""
    scl = kp["size"] * 0.5 / (1 << kp["octave"]).float()
    radii = torch.round(ORI_RADIUS * scl).long()
    sigmas = ORI_SIG * scl
    n = len(radii)
    hist = torch.zeros((n, ORI_BINS), device=scl.device)
    edges = [-1, 8, 12, 10**9]
    for low, high in zip(edges, edges[1:]):
        sel = torch.nonzero((radii > low) & (radii <= high))[:, 0]
        if len(sel) == 0:
            continue
        R = int(radii[sel].max())
        dx, dy, inside, offs = _window(
            gauss, kp["octave"][sel], kp["layer"][sel], kp["r"][sel], kp["c"][sel], R
        )
        rad = radii[sel][:, None, None]
        within = (offs.abs()[None, :, None] <= rad) & (offs.abs()[None, None, :] <= rad)
        d2 = (offs[None, :, None] ** 2 + offs[None, None, :] ** 2).float()
        w = torch.exp(d2 * (-1.0 / (2.0 * sigmas[sel] * sigmas[sel]))[:, None, None])
        ori = _atan2_degrees(dy, dx)
        mag = torch.sqrt(dx * dx + dy * dy)
        bins = torch.round(ori * (ORI_BINS / 360.0)).long()
        bins = torch.where(bins >= ORI_BINS, bins - ORI_BINS, bins)
        bins = torch.where(bins < 0, bins + ORI_BINS, bins)
        value = torch.where(inside & within, w * mag, torch.zeros_like(mag))
        # Fixed-order sums per bin (deterministic, unlike float scatter atomics).
        bins, value = bins.reshape(len(sel), -1), value.reshape(len(sel), -1)
        hist[sel] = torch.stack(
            [torch.where(bins == k, value, 0.0).sum(1) for k in range(ORI_BINS)], 1
        )

    def roll(shift):
        return torch.roll(hist, shift, dims=1)

    smooth = (roll(2) + roll(-2)) * (1.0 / 16.0) + (roll(1) + roll(-1)) * (4.0 / 16.0)
    smooth = smooth + hist * (6.0 / 16.0)
    omax = smooth.max(1).values
    left, right = torch.roll(smooth, 1, 1), torch.roll(smooth, -1, 1)
    peak = (smooth > left) & (smooth > right) & (smooth >= (omax * ORI_PEAK)[:, None])
    rows, js = torch.nonzero(peak, as_tuple=True)
    hl, hj, hr = left[rows, js], smooth[rows, js], right[rows, js]
    b = js.float() + 0.5 * (hl - hr) / (hl - 2 * hj + hr)
    b = torch.where(b < 0, ORI_BINS + b, torch.where(b >= ORI_BINS, b - ORI_BINS, b))
    angle = 360.0 - (360.0 / ORI_BINS) * b
    angle = torch.where((angle - 360.0).abs() < FLT_EPS, torch.zeros_like(angle), angle)
    return dict({k: v[rows] for k, v in kp.items()}, angle=angle)


def _descriptor_group(gauss: _Packed, kp: dict, radius: int) -> torch.Tensor:
    """calcSIFTDescriptor for keypoints whose window radius is at most ``radius``."""
    n = len(kp["x"])
    scale = (1 << kp["octave"]).float()
    ptx, pty = kp["x"] / scale, kp["y"] / scale
    scl = kp["size"] / scale * 0.5
    ori = 360.0 - kp["angle"]
    ori = torch.where((ori - 360.0).abs() < FLT_EPS, torch.zeros_like(ori), ori)
    hist_width = DESCR_SCL * scl
    cx, cy = torch.round(ptx).long(), torch.round(pty).long()
    angle = ori * (math.pi / 180.0)
    cos_t = (torch.cos(angle) / hist_width)[:, None, None]
    sin_t = (torch.sin(angle) / hist_width)[:, None, None]
    dx, dy, inside, offs = _window(gauss, kp["octave"], kp["layer"], cy, cx, radius)
    i = offs[None, :, None].float()
    j = offs[None, None, :].float()
    c_rot = j * cos_t - i * sin_t
    r_rot = j * sin_t + i * cos_t
    rb = r_rot + D / 2 - 0.5
    cb = c_rot + D / 2 - 0.5
    rad = kp["radius"][:, None, None]
    within = (offs.abs()[None, :, None] <= rad) & (offs.abs()[None, None, :] <= rad)
    ok = inside & within & (rb > -1) & (rb < D) & (cb > -1) & (cb < D)
    w = torch.exp((c_rot * c_rot + r_rot * r_rot) * (-1.0 / (D * D * 0.5)))
    o = _atan2_degrees(dy, dx)
    mag = torch.sqrt(dx * dx + dy * dy) * w
    obin = (o - ori[:, None, None]) * (NB / 360.0)
    r0, c0, o0 = torch.floor(rb), torch.floor(cb), torch.floor(obin)
    rf, cf, of = rb - r0, cb - c0, obin - o0
    o0 = o0.long()
    o0 = torch.where(o0 < 0, o0 + NB, o0)
    o0 = torch.where(o0 >= NB, o0 - NB, o0)
    v_r1 = mag * rf
    v_r0 = mag - v_r1
    v_rc11 = v_r1 * cf
    v_rc10 = v_r1 - v_rc11
    v_rc01 = v_r0 * cf
    v_rc00 = v_r0 - v_rc01
    # Trilinear accumulation as one batched product, (cells x samples) @ (samples x bins):
    # each sample writes 4 distinct spatial cells and 2 distinct orientation bins, so the
    # per-sample rows are built without accumulation and the sum over samples is a matmul
    # (deterministic, unlike float scatter atomics).
    cells = D + 2
    cell = (r0.long().clamp(-1, D - 1) + 1) * cells + c0.long().clamp(-1, D - 1) + 1
    spatial = torch.stack([cell, cell + 1, cell + cells, cell + cells + 1], -1)
    weights = torch.stack([v_rc00, v_rc01, v_rc10, v_rc11], -1)
    weights = torch.where(ok[..., None], weights, torch.zeros_like(weights))
    S = cell[0].numel()
    A = torch.zeros((n, S, cells * cells), device=dx.device)
    A.scatter_(2, spatial.reshape(n, S, 4), weights.reshape(n, S, 4))
    orient = torch.zeros((n, S, NB + 2), device=dx.device)
    of = of.reshape(n, S, 1)
    orient.scatter_(
        2,
        torch.cat([o0.reshape(n, S, 1), o0.reshape(n, S, 1) + 1], -1),
        torch.cat([1 - of, of], -1),
    )
    hist = torch.bmm(A.transpose(1, 2), orient)
    h = hist.view(n, cells, cells, NB + 2)[:, 1 : D + 1, 1 : D + 1]
    raw = h[..., :NB].clone()
    raw[..., 0] += h[..., NB]
    raw[..., 1] += h[..., NB + 1]
    raw = raw.reshape(n, -1)
    thr = torch.sqrt((raw * raw).sum(1)) * DESCR_MAG
    raw = torch.minimum(raw, thr[:, None])
    nrm = INT_FCTR / torch.clamp(torch.sqrt((raw * raw).sum(1)), min=FLT_EPS)
    return torch.clamp(torch.round(raw * nrm[:, None]), 0, 255)


def _descriptors(gauss: _Packed, kp: dict) -> torch.Tensor:
    """Descriptors of all keypoints, in a few window-radius buckets."""
    n = len(kp["x"])
    out = torch.zeros((n, D * D * NB), device=kp["x"].device)
    if n == 0:
        return out
    scl = kp["size"] / (1 << kp["octave"]).float() * 0.5
    radii = torch.round(DESCR_SCL * scl * 1.4142135623730951 * (D + 1) * 0.5).long()
    kp = dict(kp, radius=radii)
    edges = [-1, 20, 28, 40, 10**9]
    for low, high in zip(edges, edges[1:]):
        sel = torch.nonzero((radii > low) & (radii <= high))[:, 0]
        if len(sel) == 0:
            continue
        R = int(radii[sel].max())
        for chunk in torch.split(sel, max(1, 1_500_000 // (2 * R + 1) ** 2)):
            out[chunk] = _descriptor_group(gauss, {k: v[chunk] for k, v in kp.items()}, R)
    return out


@torch.no_grad()
def detect_and_compute(gray: np.ndarray, mask: np.ndarray | None = None, nfeatures: int = 4000):
    """(keypoints, descriptors) as ``cv2.SIFT_create(nfeatures).detectAndCompute``."""
    device = _device()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False  # restored below
    try:
        return _detect_and_compute(gray, mask, nfeatures, device)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32


def _detect_and_compute(gray, mask, nfeatures, device):
    with torch.backends.cudnn.flags(
        enabled=True, benchmark=False, deterministic=True, allow_tf32=False
    ):
        g = torch.from_numpy(np.ascontiguousarray(gray)).to(device)
        gauss_stacks, dog_stacks = _pyramid(g)
        threshold = math.floor(0.5 * CONTRAST / N_LAYERS * 255)
        cands, octaves = [], []
        for o, dog in enumerate(dog_stacks):
            cand = _extrema(dog, threshold)
            cands.append(cand)
            octaves.append(torch.full((len(cand),), o, dtype=torch.long, device=device))
        cand = torch.cat(cands)
        if len(cand) == 0:
            return [], None
        gauss = _Packed(gauss_stacks)
        kp = _refine(_Packed(dog_stacks), torch.cat(octaves), cand)
        if len(kp["x"]) == 0:
            return [], None
        kp = _orientations(gauss, kp)
        # removeDuplicatedSorted: identical (pt, size, angle)
        key = torch.stack([kp["x"], kp["y"], kp["size"], kp["angle"]], 1)
        _, first = np.unique(key.cpu().numpy(), axis=0, return_index=True)
        first = torch.from_numpy(np.sort(first)).to(device)
        kp = {k: v[first] for k, v in kp.items()}
        # retainBest with ties at the n-th response
        if nfeatures > 0 and len(kp["response"]) > nfeatures:
            ambient = torch.topk(kp["response"], nfeatures).values[-1]
            keep = kp["response"] >= ambient
            kp = {k: v[keep] for k, v in kp.items()}
        order = torch.argsort(kp["response"], descending=True, stable=True)
        kp = {k: v[order] for k, v in kp.items()}
        descriptors = _descriptors(gauss, kp)
        x = (kp["x"] * 0.5).cpu().numpy()
        y = (kp["y"] * 0.5).cpu().numpy()
        size = (kp["size"] * 0.5).cpu().numpy()
        angle = kp["angle"].cpu().numpy()
        response = kp["response"].cpu().numpy()
        packed_octave = (
            (kp["octave"] - 1).bitwise_and(255)
            + (kp["layer"] << 8)
            + (torch.round((kp["xi"] + 0.5) * 255).long() << 16)
        )
        octave = packed_octave.cpu().numpy()
        desc = descriptors.cpu().numpy().astype(np.float32)
    if mask is not None:
        h, w = mask.shape[:2]
        yi = np.clip((y + 0.5).astype(int), 0, h - 1)
        xi = np.clip((x + 0.5).astype(int), 0, w - 1)
        keep = mask[yi, xi] != 0
        x, y, size, angle, response, octave, desc = (
            a[keep] for a in (x, y, size, angle, response, octave, desc)
        )
    keypoints = [
        cv2.KeyPoint(float(a), float(b), float(s), float(t), float(rs), int(oc))
        for a, b, s, t, rs, oc in zip(x, y, size, angle, response, octave, strict=True)
    ]
    return keypoints, (desc if len(desc) else None)


@torch.no_grad()
def ratio_matches(query: np.ndarray, train: np.ndarray, ratio: float = 0.70) -> np.ndarray:
    """``BFMatcher().knnMatch(query, train, k=2)`` + Lowe ratio test, as (queryIdx, trainIdx) rows.

    SIFT descriptors are integers in 0..255, so every squared distance is an integer below 2^24.
    It is computed exactly in float32 as ``|q|^2 + |t|^2 - 2 q.t`` (TF32 off, any summation
    order), and its square root is IEEE-rounded. Distances, nearest indices (first index on
    ties, as the brute-force scan) and the ratio decisions therefore equal OpenCV's.
    """
    if len(train) < 2 or len(query) == 0:
        return np.empty((0, 2), dtype=np.int64)
    device = _device()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        q = torch.from_numpy(np.ascontiguousarray(query, np.float32)).to(device)
        t = torch.from_numpy(np.ascontiguousarray(train, np.float32)).to(device)
        squared = (q * q).sum(1)[:, None] + (t * t).sum(1)[None, :] - 2.0 * (q @ t.T)
        distance = torch.sqrt(torch.clamp(squared, min=0.0))
        best = torch.argmin(distance, 1)
        first = distance.gather(1, best[:, None])[:, 0]
        distance.scatter_(1, best[:, None], float("inf"))
        second = distance.min(1).values
        best, first, second = (x.cpu().numpy() for x in (best, first, second))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    keep = first.astype(np.float64) < ratio * second.astype(np.float64)
    return np.stack([np.flatnonzero(keep), best[keep]], 1)
