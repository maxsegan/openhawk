"""SIFT(4000) with CUDA keypoint and descriptor kernels (``sift_kernel.cu``) on torch_sift's pyramid.

The PyTorch port (``torch_sift``) issues ~4,700 small kernels per 1080p frame. It is host-bound
at ~40-60 ms and ~25 GPU-ms per frame, against ~0.1 ms of arithmetic. This keeps its pyramid,
deduplication and retainBest, and runs OpenCV's per-keypoint loops as CUDA threads:

- one thread per candidate pixel: extremum test, adjustLocalExtrema, calcOrientationHist;
- one thread per keypoint: calcSIFTDescriptor.

Each loop runs in OpenCV's sample order, accumulates sequentially and uses OpenCV's fastAtan2,
so results are deterministic. Keypoints are canonically sorted (octave, layer, row, column,
peak) before selection, so the integer append order does not matter.

    from cv.experiments.gpu_court.sift_cuda import detect_and_compute
"""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from cv.experiments.gpu_court import dt_cuda, torch_sift

SOURCE = Path(__file__).with_name("sift_kernel.cu")
RECORD = np.dtype(
    [("x", "f4"), ("y", "f4"), ("size", "f4"), ("response", "f4"), ("angle", "f4"), ("xi", "f4")]
    + [("octave", "i4"), ("layer", "i4"), ("r", "i4"), ("c", "i4"), ("peak", "i4")]
)
_LIBRARY = None
_GRAPHS: dict[tuple, tuple] = {}


def _pyramid(g: torch.Tensor):
    """torch_sift._pyramid, captured once per (device, H, W) as a CUDA graph and replayed.

    Replaying runs the same kernels on the same static buffers, so the pyramid is bit-identical
    to the eager one (checked in the tests). Only the ~200 Python launches per frame are saved.
    """
    key = (str(g.device), *g.shape)
    if key not in _GRAPHS:
        static = torch.empty_like(g)
        static.copy_(g)
        stream = torch.cuda.Stream(device=g.device)
        stream.wait_stream(torch.cuda.current_stream(g.device))
        with torch.cuda.stream(stream):
            for _ in range(2):  # warm up cudnn/allocator outside the capture
                torch_sift._pyramid(static)
        torch.cuda.current_stream(g.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gauss, dog = torch_sift._pyramid(static)
            gauss = [x.contiguous() for x in gauss]
            dog = [x.contiguous() for x in dog]
        _GRAPHS[key] = (graph, static, gauss, dog)
    graph, static, gauss, dog = _GRAPHS[key]
    static.copy_(g)
    graph.replay()
    return gauss, dog


def library():
    global _LIBRARY
    if _LIBRARY is None:
        lib = dt_cuda.build(SOURCE, "sift_kernel", ("--fmad=false",))
        p = ctypes.c_void_p
        lib.sift_detect.restype = ctypes.c_int
        lib.sift_detect.argtypes = [p, p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.sift_detect.argtypes += [ctypes.c_float] * 4 + [p, p, ctypes.c_int]
        lib.sift_descriptors.restype = ctypes.c_int
        lib.sift_descriptors.argtypes = [p, ctypes.c_int, p, p, p, p]
        if lib.keypoint_size() != RECORD.itemsize:
            raise RuntimeError("Keypoint layout differs between CUDA and NumPy")
        _LIBRARY = lib
    return _LIBRARY


def _launch(lib, g, d, o, threshold, capacity, device):
    H, W = d.shape[1:]
    out = torch.empty(capacity * RECORD.itemsize, dtype=torch.uint8, device=device)
    count = torch.zeros(1, dtype=torch.int32, device=device)
    rc = lib.sift_detect(
        d.data_ptr(),
        g.data_ptr(),
        H,
        W,
        o,
        float(threshold),
        torch_sift.CONTRAST,
        torch_sift.EDGE,
        torch_sift.SIGMA,
        out.data_ptr(),
        count.data_ptr(),
        capacity,
    )
    if rc:
        raise RuntimeError("SIFT detect kernel failed")
    return out, count


def _detect(gauss, dog, device) -> np.ndarray:
    """All octaves launched back to back, then one synchronising read of the counts."""
    lib = library()
    threshold = math.floor(0.5 * torch_sift.CONTRAST / torch_sift.N_LAYERS * 255)
    launched = {}
    for o, (g, d) in enumerate(zip(gauss, dog, strict=True)):
        H, W = d.shape[1:]
        if H <= 2 * torch_sift.BORDER or W <= 2 * torch_sift.BORDER:
            continue
        capacity = max(1 << 10, 1 << 16 >> (2 * o))
        launched[o] = (capacity, *_launch(lib, g, d, o, threshold, capacity, device))
    if not launched:
        return np.empty(0, RECORD)
    # one transfer: every octave's counts and record buffers together
    flat = torch.cat(
        [c.view(torch.uint8) for _, _, c in launched.values()]
        + [o for _, o, _ in launched.values()]
    )
    host = flat.cpu().numpy()
    counts = host[: 4 * len(launched)].view(np.int32)
    offset = 4 * len(launched)
    found = []
    for (o, (capacity, out, _)), n in zip(launched.items(), counts, strict=True):
        n = int(n)
        block = host[offset : offset + capacity * RECORD.itemsize]
        offset += capacity * RECORD.itemsize
        if n > capacity:  # rare: rerun this octave with room for every keypoint
            capacity = 1 << int(math.ceil(math.log2(n)))
            out, count = _launch(lib, gauss[o], dog[o], o, threshold, capacity, device)
            n = int(count.item())
            block = out.cpu().numpy()
        found.append(block[: n * RECORD.itemsize].view(RECORD))
    if not found:
        return np.empty(0, RECORD)
    records = np.concatenate(found)
    order = np.lexsort(
        (records["peak"], records["c"], records["r"], records["layer"], records["octave"])
    )
    return records[order]


@torch.no_grad()
def detect_and_compute(gray: np.ndarray, mask: np.ndarray | None = None, nfeatures: int = 4000):
    """(keypoints, descriptors) as ``cv2.SIFT_create(nfeatures).detectAndCompute``."""
    device = torch_sift._device()
    with torch.backends.cudnn.flags(
        enabled=True, benchmark=False, deterministic=True, allow_tf32=False
    ):
        torch.cuda.set_device(device)
        g = torch.from_numpy(np.ascontiguousarray(gray)).to(device)
        gauss, dog = _pyramid(g)
        records = _detect(gauss, dog, device)
        if len(records) == 0:
            return [], None
        # removeDuplicatedSorted (identical pt, size, angle): first in canonical order
        # first of each identical (x, y, size, angle) group, in canonical order (np.unique's
        # return_index semantics without its row sort)
        order = np.lexsort((records["angle"], records["size"], records["y"], records["x"]))
        ordered = records[order]
        same = np.ones(len(order), bool)
        for k in ("x", "y", "size", "angle"):
            same[1:] &= ordered[k][1:] == ordered[k][:-1]
        same[0] = False
        records = records[np.sort(order[~same])]
        if nfeatures > 0 and len(records) > nfeatures:
            ambient = np.partition(records["response"], len(records) - nfeatures)[
                len(records) - nfeatures
            ]
            records = records[records["response"] >= ambient]
        records = records[np.argsort(-records["response"], kind="stable")]
        kps = torch.from_numpy(records.view(np.uint8).copy()).to(device)
        pointers = torch.tensor([x.data_ptr() for x in gauss], dtype=torch.int64, device=device)
        heights = torch.tensor([x.shape[1] for x in gauss], dtype=torch.int32, device=device)
        widths = torch.tensor([x.shape[2] for x in gauss], dtype=torch.int32, device=device)
        out = torch.empty((len(records), 128), dtype=torch.float32, device=device)
        rc = library().sift_descriptors(
            kps.data_ptr(),
            len(records),
            pointers.data_ptr(),
            heights.data_ptr(),
            widths.data_ptr(),
            out.data_ptr(),
        )
        if rc:
            raise RuntimeError("SIFT descriptor kernel failed")
        desc = out.cpu().numpy()
    x, y = records["x"] * 0.5, records["y"] * 0.5
    size, angle, response = records["size"] * 0.5, records["angle"], records["response"]
    octave = (
        ((records["octave"] - 1) & 255)
        + (records["layer"] << 8)
        + (np.rint((records["xi"] + 0.5) * 255).astype(np.int64) << 16)
    )
    if mask is not None:
        h, w = mask.shape[:2]
        keep = (
            mask[np.clip((y + 0.5).astype(int), 0, h - 1), np.clip((x + 0.5).astype(int), 0, w - 1)]
            != 0
        )
        x, y, size, angle, response, octave, desc = (
            a[keep] for a in (x, y, size, angle, response, octave, desc)
        )
    keypoints = [
        cv2.KeyPoint(float(a), float(b), float(s), float(t), float(rs), int(oc))
        for a, b, s, t, rs, oc in zip(x, y, size, angle, response, octave, strict=True)
    ]
    return keypoints, (desc if len(desc) else None)
