"""GPU JPEG decode that reproduces ``cv2.imread`` byte for byte (throughput only).

nvJPEG's own RGB output differs from libjpeg-turbo (the decoder inside OpenCV) in chroma
upsampling and colour conversion: luma is identical, but about 14% of RGB samples move, some
by tens of levels at colour edges. This module asks nvJPEG only for what matches, the planar
Y/Cb/Cr samples after the inverse DCT, and redoes the two libjpeg-turbo steps on the GPU with
libjpeg-turbo's integer arithmetic:

- ``h2v2_fancy_upsample`` (jdsample.c): the triangle filter with alternating +8/+7 rounding,
  edge rows and columns replicated, as libjpeg's context rows do;
- ``ycc_rgb_convert`` (jdcolor.c): the 16-bit fixed-point tables and range limiting.

The result equals ``cv2.imread`` exactly for 4:2:0 and 4:4:4 baseline JPEGs (what ffmpeg's
``-q:v 2`` frame extraction writes). Anything else -- other subsampling, a decode error, no
nvJPEG library -- falls back to ``cv2.imread`` for that picture, so a caller never gets a
different picture, only a slower one. ``verify=True`` compares a sample with ``cv2.imread``.

The default backend is nvJPEG's hardware decoder (the NVJPG engines on Ampere A100/H100 and
Blackwell RTX PRO cards), which leaves the SMs to the models; ``backend="default"`` uses the
CUDA/hybrid path on cards without the engines.

    decoder = NvJpegDecoder(device=0)
    bgr = decoder.decode_paths(paths)              # list of HxWx3 uint8 numpy arrays (== cv2.imread)
    rgb = decoder.decode_paths_rgb_chw(paths)      # (N, 3, H, W) uint8 CUDA tensor
"""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

BACKENDS = {"default": 0, "hybrid": 1, "gpu_hybrid": 2, "hardware": 3}
OUTPUT_YUV = 1
# nvjpegChromaSubsampling_t
CSS_444, CSS_420 = 0, 2


class _Image(ctypes.Structure):
    _fields_ = [("channel", ctypes.c_void_p * 4), ("pitch", ctypes.c_size_t * 4)]


def _library_candidates() -> list[str]:
    found = []
    if os.environ.get("TENNIS_NVJPEG_LIBRARY"):
        found.append(os.environ["TENNIS_NVJPEG_LIBRARY"])
    name = ctypes.util.find_library("nvjpeg")
    if name:
        found.append(name)
    found.extend(sorted(glob.glob("/usr/local/cuda*/lib64/libnvjpeg.so.12*")))
    try:  # pip wheels: nvidia-nvjpeg-cu12, or the copy torchvision bundles
        import torchvision

        root = os.path.dirname(os.path.dirname(torchvision.__file__))
        found.extend(sorted(glob.glob(os.path.join(root, "nvidia", "nvjpeg", "lib", "libnvjpeg*"))))
        found.extend(sorted(glob.glob(os.path.join(root, "torchvision.libs", "libnvjpeg*"))))
    except ImportError:
        pass
    return found


def load_library() -> tuple[ctypes.CDLL, str]:
    errors = []
    for path in _library_candidates():
        try:
            return ctypes.CDLL(path), path
        except OSError as error:
            errors.append(f"{path}: {error}")
    raise OSError("no loadable libnvjpeg: " + "; ".join(errors or ["no candidates"]))


def _check(status: int, what: str) -> None:
    if status != 0:
        raise RuntimeError(f"nvJPEG {what} failed with status {status}")


# ---------------------------------------------------------------------------------------------
# libjpeg-turbo arithmetic on the GPU


def _fix(value: float) -> int:
    """jdcolor.c FIX(x) with SCALEBITS 16."""
    return int(value * (1 << 16) + 0.5)


def _ycc_tables(torch, device):
    """jdcolor.c build_ycc_rgb_table, evaluated for x = sample - 128 in int32 (all terms
    stay below 2^24, so the table lookups become exact elementwise arithmetic)."""
    return _fix(1.40200), _fix(1.77200), -_fix(0.71414), -_fix(0.34414)


def fancy_upsample_h2v2(plane, torch):
    """jdsample.c h2v2_fancy_upsample on (N, h, w) uint8 -> (N, 2h, 2w) uint8."""
    c = plane.to(torch.int16)
    above = torch.cat([c[:, :1], c[:, :-1]], dim=1)
    below = torch.cat([c[:, 1:], c[:, -1:]], dim=1)
    rows = []
    for colsum in (3 * c + above, 3 * c + below):  # output row 2i (nearer above), 2i+1 (below)
        left = torch.cat([colsum[..., :1], colsum[..., :-1]], dim=2)
        right = torch.cat([colsum[..., 1:], colsum[..., -1:]], dim=2)
        even = (3 * colsum + left + 8) >> 4  # first column: 4 * colsum + 8
        odd = (3 * colsum + right + 7) >> 4  # last column: 4 * colsum + 7
        rows.append(torch.stack([even, odd], dim=-1).flatten(-2))
    n, h, w2 = rows[0].shape
    return torch.stack(rows, dim=2).reshape(n, 2 * h, w2).to(torch.uint8)


def ycc_to_rgb(y, cb, cr, tables, torch):
    """jdcolor.c ycc_rgb_convert on full-resolution (N, H, W) uint8 planes -> (N, 3, H, W).

    R = y + ((FIX(1.402) x_cr + 2^15) >> 16), B = y + ((FIX(1.772) x_cb + 2^15) >> 16),
    G = y + ((-FIX(0.34414) x_cb + 2^15 - FIX(0.71414) x_cr) >> 16), range-limited; ``>>``
    is libjpeg's arithmetic RIGHT_SHIFT."""
    cr_r, cb_b, cr_g, cb_g = tables
    half = 1 << 15
    x_cb = cb.to(torch.int32) - 128
    x_cr = cr.to(torch.int32) - 128
    yi = y.to(torch.int32)
    out = torch.empty((y.shape[0], 3, *y.shape[1:]), dtype=torch.uint8, device=y.device)
    out[:, 0] = (yi + ((cr_r * x_cr + half) >> 16)).clamp_(0, 255)
    out[:, 1] = (yi + ((cb_g * x_cb + half + cr_g * x_cr) >> 16)).clamp_(0, 255)
    out[:, 2] = (yi + ((cb_b * x_cb + half) >> 16)).clamp_(0, 255)
    return out


def use_blocking_sync(device: int) -> bool:
    """Make host threads sleep, not spin, while they wait for this device (process-wide).

    CUDA's default schedule spins a core in every synchronising call; with decode moved to
    the GPU, those waits are most of what the decode threads do. Sets
    CU_CTX_SCHED_BLOCKING_SYNC on the primary context (allowed while active since CUDA 11).
    Throughput and CPU only; results are unchanged. Returns False when it could not be set.
    """
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        handle = ctypes.c_int()
        if cuda.cuInit(0) or cuda.cuDeviceGet(ctypes.byref(handle), device):
            return False
        return cuda.cuDevicePrimaryCtxSetFlags(handle, 0x04) == 0
    except OSError:
        return False


# ---------------------------------------------------------------------------------------------


@dataclass
class DecodeStats:
    gpu: int = 0
    fallback: int = 0
    fallback_reasons: dict = field(default_factory=dict)

    def fell_back(self, reason: str) -> None:
        self.fallback += 1
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1


class NvJpegDecoder:
    """Batched nvJPEG decode to libjpeg-turbo-identical pictures on one CUDA device."""

    def __init__(
        self,
        device: int = 0,
        batch_size: int = 32,
        backend: str = "hardware",
        blocking_sync: bool = True,
    ):
        import torch

        self.torch = torch
        self.device = torch.device(f"cuda:{device}")
        # Without it every wait for a decoded batch spins a core, which is most of the CPU
        # the GPU decode would otherwise free (see use_blocking_sync).
        self.blocking_sync = blocking_sync and use_blocking_sync(device)
        with torch.cuda.device(self.device):
            torch.zeros(1, device=self.device)  # primary context, shared with nvJPEG
        self.lib, self.library_path = load_library()
        self.backend = backend
        self.batch_size = batch_size
        self.handle = ctypes.c_void_p()
        _check(
            self.lib.nvjpegCreateEx(BACKENDS[backend], None, None, 0, ctypes.byref(self.handle)),
            f"create ({backend} backend)",
        )
        self.state = ctypes.c_void_p()
        _check(self.lib.nvjpegJpegStateCreate(self.handle, ctypes.byref(self.state)), "state")
        self._initialised_batch = 0
        self.tables = _ycc_tables(torch, self.device)
        self.stats = DecodeStats()

    def identity(self) -> dict:
        return {
            "decoder": "gpu_jpeg.NvJpegDecoder",
            "library": self.library_path,
            "backend": self.backend,
            "blocking_sync": self.blocking_sync,
            "contract": "libjpeg-turbo h2v2 fancy upsampling + ycc_rgb tables on nvJPEG YUV",
        }

    def _info(self, data: bytes) -> tuple[int, int, list[int], list[int]]:
        components, subsampling = ctypes.c_int(), ctypes.c_int()
        widths, heights = (ctypes.c_int * 4)(), (ctypes.c_int * 4)()
        _check(
            self.lib.nvjpegGetImageInfo(
                self.handle,
                data,
                len(data),
                ctypes.byref(components),
                ctypes.byref(subsampling),
                widths,
                heights,
            ),
            "image info",
        )
        return components.value, subsampling.value, list(widths), list(heights)

    def _decode_group(self, datas: list[bytes], widths: list[int], heights: list[int]):
        """One nvjpegDecodeBatched call over same-geometry pictures -> Y, Cb, Cr tensors."""
        torch = self.torch
        count = len(datas)
        if count != self._initialised_batch:
            _check(
                self.lib.nvjpegDecodeBatchedInitialize(
                    self.handle, self.state, count, 1, OUTPUT_YUV
                ),
                "batched initialise",
            )
            self._initialised_batch = count
        planes = [
            torch.empty((count, heights[i], widths[i]), dtype=torch.uint8, device=self.device)
            for i in range(3)
        ]
        images = (_Image * count)()
        for b in range(count):
            for i in range(3):
                images[b].channel[i] = planes[i][b].data_ptr()
                images[b].pitch[i] = widths[i]
        pointers = (ctypes.c_char_p * count)(*datas)
        lengths = (ctypes.c_size_t * count)(*[len(d) for d in datas])
        current = torch.cuda.current_stream(self.device)
        stream = ctypes.c_void_p(current.cuda_stream)
        # The hardware decoder does not wait for work already queued on the stream. Without
        # this sync, a second same-size batch in one call can overwrite memory that the
        # previous batch's colour-conversion kernels are still reading, and those pictures come
        # back corrupted (seen with 64-frame calls: the first 4 of the call, nondeterministic).
        current.synchronize()
        _check(
            self.lib.nvjpegDecodeBatched(
                self.handle, self.state, pointers, lengths, images, stream
            ),
            "batched decode",
        )
        current.synchronize()  # the inputs above are host objects freed on return
        return planes

    def _to_rgb(self, y, cb, cr, subsampling: int, height: int, width: int):
        torch = self.torch
        if subsampling == CSS_420:
            cb = fancy_upsample_h2v2(cb, torch)[:, :height, :width]
            cr = fancy_upsample_h2v2(cr, torch)[:, :height, :width]
        return ycc_to_rgb(y, cb, cr, self.tables, torch)

    def decode_bytes_rgb_chw(self, datas: list[bytes], names: list[str] | None = None):
        """(N, 3, H, W) uint8 RGB on the device; pictures must share one geometry."""
        torch = self.torch
        names = names or [str(i) for i in range(len(datas))]
        out: list = [None] * len(datas)
        groups: dict[tuple, list[int]] = {}
        for index, data in enumerate(datas):
            try:
                components, subsampling, widths, heights = self._info(data)
            except RuntimeError:
                components, subsampling = 0, -1
            if components != 3 or subsampling not in (CSS_420, CSS_444):
                out[index] = "fallback"
                self.stats.fell_back(f"components={components} subsampling={subsampling}")
                continue
            key = (subsampling, tuple(widths[:3]), tuple(heights[:3]))
            groups.setdefault(key, []).append(index)
        with torch.cuda.device(self.device), torch.inference_mode():
            for (subsampling, widths, heights), members in groups.items():
                for start in range(0, len(members), self.batch_size):
                    chunk = members[start : start + self.batch_size]
                    try:
                        y, cb, cr = self._decode_group(
                            [datas[i] for i in chunk], list(widths), list(heights)
                        )
                    except RuntimeError as error:
                        for i in chunk:
                            out[i] = "fallback"
                            self.stats.fell_back(str(error))
                        continue
                    rgb = self._to_rgb(y, cb, cr, subsampling, heights[0], widths[0])
                    for position, i in enumerate(chunk):
                        out[i] = rgb[position]
                    self.stats.gpu += len(chunk)
            for i, item in enumerate(out):
                if isinstance(item, str):
                    bgr = cv2.imdecode(np.frombuffer(datas[i], np.uint8), cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise OSError(f"cannot decode {names[i]}")
                    out[i] = torch.from_numpy(bgr[:, :, ::-1].transpose(2, 0, 1).copy()).to(
                        self.device
                    )
            shapes = {tuple(t.shape) for t in out}
            if len(shapes) != 1:
                raise ValueError(f"mixed picture geometries in one batch: {sorted(shapes)}")
            return torch.stack(out)

    def decode_paths_rgb_chw(self, paths: list[str]):
        datas = [_read(path) for path in paths]
        return self.decode_bytes_rgb_chw(datas, [os.fspath(p) for p in paths])

    def decode_paths(self, paths: list[str]) -> list[np.ndarray]:
        """HxWx3 BGR uint8 host arrays equal to ``cv2.imread(path)`` (pinned host memory)."""
        if not paths:
            return []
        torch = self.torch
        with torch.cuda.device(self.device):
            rgb = self.decode_paths_rgb_chw(paths)
            bgr = rgb.flip(1).permute(0, 2, 3, 1).contiguous()
            host = torch.empty(bgr.shape, dtype=torch.uint8, pin_memory=True)
            host.copy_(bgr, non_blocking=True)
            torch.cuda.current_stream(self.device).synchronize()
        return list(host.numpy())


def open_decoder(device: int, **options) -> "NvJpegDecoder | None":
    """An NvJpegDecoder, or None (callers then decode with cv2.imread) where it cannot run.

    Cards without NVJPG engines (consumer GeForce such as the 5090/3090) refuse the hardware
    backend, and the other backends are not byte-identical to cv2.imread. Pictures are the
    same either way, so falling back changes throughput only; it is reported on stderr.
    """
    try:
        return NvJpegDecoder(device, **options)
    except (OSError, RuntimeError) as error:
        print(
            f"gpu_jpeg: nvjpeg unavailable on cuda:{device} ({error}); decoding with cv2",
            file=sys.stderr,
            flush=True,
        )
        return None


def _read(path) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def verify_against_cv2(decoder: NvJpegDecoder, paths: list[str]) -> dict:
    """Byte comparison of the GPU decode with ``cv2.imread`` over ``paths``."""
    different = 0
    worst = 0
    for start in range(0, len(paths), decoder.batch_size):
        chunk = paths[start : start + decoder.batch_size]
        for path, gpu in zip(chunk, decoder.decode_paths(chunk), strict=True):
            cpu = cv2.imread(os.fspath(path))
            if cpu.shape != gpu.shape or not np.array_equal(cpu, gpu):
                different += 1
                if cpu.shape == gpu.shape:
                    worst = max(worst, int(np.abs(cpu.astype(int) - gpu.astype(int)).max()))
    return {"pictures": len(paths), "different": different, "max_abs_difference": worst}
