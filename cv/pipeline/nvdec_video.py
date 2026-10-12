"""NVDEC video decode under the stages' own ffmpeg commands (throughput only, same bytes).

Several upstream stages decode the whole source video on the CPU through an ffmpeg
subprocess (``broadcast_source`` twice, ``shot_boundaries``, ``native_view_frames``) and
``frame_extraction`` decodes every point window. For AV1/H.264/VP9 the decoded 8-bit 4:2:0
samples are fixed by the codec specification, and NVDEC's planes equal libavcodec's
(600/600 frames on an AV1 pilot source, zero difference). This module therefore keeps each
stage's ffmpeg command -- filters, scaling, range conversion, JPEG encoder, showinfo clock --
and replaces only its input: the GPU decodes the video and the frames are muxed, with their
source timestamps, colour and aspect metadata, as rawvideo NUT into ``ffmpeg -f nut -i -``.
Outputs are byte-identical to the CPU path (tests: overlay, native frames, point extraction,
shot-boundary signals).

Switch: ``TENNIS_VIDEO_DECODE=nvdec`` (or ``nvdec:<gpu>``; default GPU from
``TENNIS_NVDEC_GPU``, else 0 in ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` numbering). Off by default.
Commands this module does not recognise, sources that are not 8-bit 4:2:0, a missing
PyNvVideoCodec, or a decoder that refuses the codec run the original command unchanged
(reported once on stderr), so a caller never gets different output, only a slower one.

Recognised command forms (everything else falls back):

- whole video: ``ffmpeg [-hide_banner] -nostdin [-v L] -i VIDEO <output options>``;
- one span with an accurate seek: ``... -ss S -i VIDEO [-t D] <output options>``;
- the native point extraction: ``... -seek_timestamp 1 -ss S -copyts -noaccurate_seek -i
  VIDEO -vf select=...`` (``players.native_extract_command``).

Frames are only piped where ffmpeg could use them: a chain that starts with ``fps=N`` gets the
last frame of each output tick (the only frame the fps filter can emit for that tick), and a
process that stops reading (``-frames:v``) ends the feed. One decoder per (video, GPU) is kept
per process and decodes forward across calls; a window more than SEEK_AHEAD_SECONDS ahead
(or behind) restarts at the keyframe before it.
"""

from __future__ import annotations

import itertools
import os
import re
import subprocess
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from cv.pipeline import cuda_blocking_sync

# Input-side options this module understands; anything else before ``-i`` falls back.
_INPUT_FLAGS = {"-copyts", "-noaccurate_seek", "-hide_banner", "-nostdin"}
_INPUT_VALUES = {"-ss", "-seek_timestamp", "-v", "-loglevel"}
_warned: set[str] = set()
# A point window further ahead than this seeks to the keyframe before it instead of decoding
# the gap (a forward decode of a whole 3 h match doubled frame_extraction's wall).
SEEK_AHEAD_SECONDS = 10


MIN_FREE_MIB = 2048
# A decoded picture whose pts does not increase: ``drop`` it as ffmpeg's fps filter and CFR
# output do (default), or ``stop`` (rollback: the feed ends there and the output is short).
OUT_OF_ORDER = os.environ.get("TENNIS_NVDEC_OUT_OF_ORDER", "drop")
# A CFR output whose first picture starts after the container: repeat it back to the origin as
# ffmpeg does on the CPU path (``on``, default) or not (``off``, rollback: indices 2 early at
# a 0.033 s video start).
START_PAD = os.environ.get("TENNIS_NVDEC_START_PAD", "on")


def requested_gpu() -> int | None:
    """GPU for NVDEC when the switch is on, else None."""
    value = os.environ.get("TENNIS_VIDEO_DECODE", "cpu").strip().lower()
    if not value.startswith("nvdec"):
        return None
    _, _, gpu = value.partition(":")
    return int(gpu or os.environ.get("TENNIS_NVDEC_GPU", "0"))


def _note(key: str, message: str, kind: str | None = None) -> None:
    """Warn once per key; ``kind`` marks a CPU-decode fallback (recorded per step)."""
    if key not in _warned:
        _warned.add(key)
        print(f"nvdec_video: {message}", file=sys.stderr, flush=True)
    if kind:
        from cv.pipeline import switch_fallbacks

        switch_fallbacks.record("TENNIS_VIDEO_DECODE", kind, message, where=_caller())


def _open_kind(error: BaseException) -> str:
    return "module_missing" if isinstance(error, ImportError) else "decoder_unavailable"


def _caller() -> str:
    """The stage module that asked for the decode (``shot_boundaries``, ...)."""
    frame = sys._getframe(1)
    while frame is not None and frame.f_globals.get("__name__") == __name__:
        frame = frame.f_back
    return str(frame.f_globals.get("__name__", "")).rsplit(".", 1)[-1] if frame else ""


@dataclass(frozen=True)
class SourceInfo:
    width: int
    height: int
    time_base: Fraction
    start_pts: int
    rate: Fraction
    sar: Fraction | None
    color: dict
    tags: str  # setparams options restoring what rawvideo NUT drops (range, matrix, ...)


def probe(video: str) -> SourceInfo | None:
    """Stream facts the NUT pipe must carry, or None if NVDEC cannot stand in exactly."""
    import av

    with av.open(video) as container:
        stream = container.streams.video[0]
        context = stream.codec_context
        if context.pix_fmt not in ("yuv420p", "yuvj420p"):
            return None
        start = stream.start_time if stream.start_time is not None else 0
        # ffmpeg's default input offset subtracts the container start time.
        container_start = container.start_time
        if container_start is not None:
            start = round(Fraction(container_start, 1_000_000) / stream.time_base)
        tags = _colour_tags(video)
        return SourceInfo(
            width=context.width,
            height=context.height,
            time_base=Fraction(stream.time_base),
            start_pts=int(start),
            rate=Fraction(stream.guessed_rate or stream.average_rate or 25),
            sar=Fraction(stream.sample_aspect_ratio) if stream.sample_aspect_ratio else None,
            color={
                "color_range": int(context.color_range),
                "colorspace": int(context.colorspace),
                "color_primaries": int(context.color_primaries),
                "color_trc": int(context.color_trc),
            },
            tags=tags,
        )


def _colour_tags(video: str) -> str:
    """``setparams`` options for the colour properties ffprobe reports on the first video
    stream. rawvideo NUT does not carry them, and the stages' scalers use them (a tv-range
    bt709 source goes to full-range gray through RGB, for instance)."""
    names = {
        "color_range": "range",
        "color_primaries": "color_primaries",
        "color_transfer": "color_trc",
        "color_space": "colorspace",
        "chroma_location": "chroma_location",
    }
    raw = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=" + ",".join(names),
            "-of",
            "json",
            video,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    import json

    stream = (json.loads(raw).get("streams") or [{}])[0]
    return ":".join(
        f"{option}={stream[key]}"
        for key, option in names.items()
        if stream.get(key) not in (None, "unknown", "unspecified", "reserved")
    )


class _Decoder:
    """Forward NVDEC decode of one video on one GPU; yields (pts, NV12 CUDA tensor)."""

    def __init__(self, video: str, gpu: int, info: SourceInfo):
        import PyNvVideoCodec as nvc
        import torch

        self.nvc, self.torch = nvc, torch
        _sleep_while_waiting(gpu)
        self.video, self.gpu, self.info = video, gpu, info
        self.device = torch.device("cuda", gpu)
        self._open()

    def _open(self, seek_index: int | None = None) -> None:
        nvc = self.nvc
        self.demuxer = nvc.CreateDemuxer(filename=self.video)
        self.decoder = nvc.CreateDecoder(
            gpuid=self.gpu,
            codec=self.demuxer.GetNvCodecId(),
            cudacontext=0,
            cudastream=0,
            usedevicememory=True,
        )
        self.pending: list = []  # frames of the latest Decode() call, not yet consumed
        self.last_pts: int | None = None
        self.packets = iter(self.demuxer)
        if seek_index:
            # Seek(frame index) returns the keyframe packet at or before that frame; the
            # demuxer then continues after it. Decoding from a keyframe is exact.
            first = self.demuxer.Seek(seek_index)
            self.packets = itertools.chain([first], self.packets)

    def _next(self):
        """(pts, NV12 tensor view). The view is valid only until the following call."""
        if self._peek_pts() is None:
            return None
        pts, frame = self.pending.pop(0)
        self.last_pts = pts
        return pts, self.torch.from_dlpack(frame)

    def _peek_pts(self) -> int | None:
        """pts of the next frame without consuming it (None at the end)."""
        while not self.pending:
            packet = next(self.packets, None)
            frames = self.decoder.Decode(self.nvc.PacketData() if packet is None else packet)
            if packet is None and not frames:
                return None
            self.pending = [(int(f.getPTS()), f) for f in frames]
        return self.pending[0][0]

    def frames_from(self, pts: int):
        """Frames with pts >= ``pts`` in presentation order. A target behind the current
        position, or more than ``SEEK_AHEAD_SECONDS`` ahead of it, restarts at the keyframe
        before it; if the demuxer lands after the target (seek accuracy varies by container)
        the pass restarts from the first frame instead, so no target frame is ever skipped."""
        seconds = (pts - self.info.start_pts) * self.info.time_base
        behind = self.last_pts is not None and pts <= self.last_pts
        position = self.info.start_pts if self.last_pts is None else self.last_pts
        far = (pts - position) * self.info.time_base > SEEK_AHEAD_SECONDS
        index = int(seconds * self.info.rate) - 1  # a frame at or before the target
        if (behind or far) and index > 0:
            self._open(seek_index=index)
            first = self._peek_pts()
            if first is None or first > pts:
                _note(f"seek:{self.video}", "NVDEC seek landed after its target; decoding forward")
                self._open()
        elif behind:
            self._open()
        while True:
            item = self._next()
            if item is None:
                return
            if item[0] >= pts:
                yield item


def _sleep_while_waiting(gpu: int) -> None:
    # Without CU_CTX_SCHED_BLOCKING_SYNC every wait for a decoded picture spins a core: on a
    # busy card the decode thread's CPU time equalled its wall time (18.6 s vs 1.3 s).
    # (cuda_blocking_sync.enable is opt-in for every device; NVDEC always wants it on its own.)
    import ctypes

    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        handle = ctypes.c_int()
        if cuda.cuInit(0) == 0 and cuda.cuDeviceGet(ctypes.byref(handle), gpu) == 0:
            cuda.cuDevicePrimaryCtxSetFlags(handle, cuda_blocking_sync.CU_CTX_SCHED_BLOCKING_SYNC)
    except OSError:
        pass


_decoders: dict[tuple[str, int], _Decoder] = {}


def _decoder(video: str, gpu: int, info: SourceInfo) -> _Decoder:
    key = (os.path.realpath(video), gpu)
    if key not in _decoders:
        _decoders[key] = _Decoder(video, gpu, info)
    return _decoders[key]


@dataclass
class _Plan:
    video: str
    piped: list[str]  # the ffmpeg command reading NUT from stdin
    first_pts: int  # first source pts to send
    shift: int  # subtracted from source pts (what ffmpeg's input offset would do)
    end_pts: int | None  # exclusive, in source pts
    fps: Fraction | None  # leading fps filter: send the last frame of each output tick
    # A CFR output (no fps filter, no -copyts) whose first picture lies a frame slot or more
    # after the plan's origin: ffmpeg repeats that picture back to the origin (``START_PAD``).
    pad_start: bool = False


def plan(command: list[str], info_for=probe) -> _Plan | None:
    """The piped equivalent of ``command``, or None when it is not a recognised form."""
    if not command or Path(command[0]).name != "ffmpeg" or command.count("-i") != 1:
        return None
    at = command.index("-i")
    video = command[at + 1]
    before, after = command[1:at], command[at + 2 :]
    options: dict[str, str | bool] = {}
    i = 0
    while i < len(before):
        flag = before[i]
        if flag in _INPUT_FLAGS:
            options[flag] = True
            i += 1
        elif flag in _INPUT_VALUES and i + 1 < len(before):
            options[flag] = before[i + 1]
            i += 2
        else:
            return None
    if not Path(video).is_file():
        return None
    info = info_for(video)
    if info is None:
        return None
    tb = info.time_base
    seek = float(options.get("-ss", 0.0))
    copyts = bool(options.get("-copyts"))
    if copyts and not options.get("-noaccurate_seek"):
        return None
    duration = None
    if "-t" in after:
        # Only an output -t directly after the input (shot_boundaries / view spans).
        j = after.index("-t")
        if j != 0 or copyts:
            return None
        duration = float(after[j + 1])
    if copyts:
        # -copyts -noaccurate_seek: timestamps untouched; the command's own select filter
        # drops pictures before its target, so any start at or before the seek is exact.
        first, shift = round(Fraction(seek) / tb), 0
    else:
        # Accurate seek: ffmpeg drops pictures before S and shifts by (container start + S).
        shift = info.start_pts + round(Fraction(seek) / tb)
        first = shift
    end = None if duration is None else shift + round(Fraction(duration) / tb)
    filters = None
    if "-vf" in after:
        filters = after[after.index("-vf") + 1]
    fps = None
    if filters:
        head = filters.split(",", 1)[0]
        match = re.fullmatch(r"fps=([0-9./]+)", head)
        if match:
            fps = Fraction(match.group(1))
    level = options.get("-v", options.get("-loglevel"))
    piped = ["ffmpeg"]
    if options.get("-hide_banner"):
        piped.append("-hide_banner")
    piped.append("-nostdin")
    if level:
        piped += ["-v", str(level)]
    # -copyts: ffmpeg keeps the piped timestamps, which are already shifted as the CPU path
    # would shift them; -f nut carries pts, time base, rate and colour properties.
    piped += ["-copyts", "-f", "nut", "-i", "-"]
    rest = list(after)
    if duration is not None:
        del rest[0:2]  # the span is cut by the feed
    # rawvideo NUT loses the aspect ratio and colour tags and rescales the time base; put
    # back exactly what the decoder would have attached before the command's own filters.
    prefix = [f"settb={tb.numerator}/{tb.denominator}"]
    if info.sar:
        prefix.append(f"setsar={info.sar.numerator}/{info.sar.denominator}")
    if info.tags:
        prefix.append(f"setparams={info.tags}")
    head = ",".join(prefix)
    if "-vf" in rest:
        k = rest.index("-vf") + 1
        rest[k] = f"{head},{rest[k]}"
    else:
        rest = ["-vf", head, *rest]
    piped += rest
    return _Plan(video, piped, first, shift, end, fps, pad_start=not copyts and fps is None)


def _feed(plan_: _Plan, info: SourceInfo, gpu: int, stdin) -> None:
    import av
    import torch

    decoder = _decoder(plan_.video, gpu, info)
    W, H = info.width, info.height
    size = W * H
    device_buffer = torch.empty(size * 3 // 2, dtype=torch.uint8, device=decoder.device)
    host = torch.empty(size * 3 // 2, dtype=torch.uint8, pin_memory=True)
    muxer = av.open(stdin, "w", format="nut")
    stream = muxer.add_stream("rawvideo", rate=info.rate)
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
    stream.time_base = info.time_base
    for key, value in info.color.items():
        setattr(stream.codec_context, key, value)
    frame_ticks = max(1, round(1 / (info.rate * info.time_base)))
    tick = None
    held = None  # (pts, tensor) waiting to learn whether it ends its fps tick

    def send(pts: int, nv12) -> None:
        with torch.cuda.device(decoder.device):
            device_buffer[:size].view(H, W).copy_(nv12[:H])
            uv = nv12[H : H + H // 2].view(H // 2, W // 2, 2)
            device_buffer[size : size + size // 4].view(H // 2, W // 2).copy_(uv[..., 0])
            device_buffer[size + size // 4 :].view(H // 2, W // 2).copy_(uv[..., 1])
            host.copy_(device_buffer)
        packet = av.Packet(host.numpy())
        packet.pts = packet.dts = pts - plan_.shift
        packet.duration = frame_ticks
        packet.time_base = info.time_base
        packet.stream = stream
        muxer.mux(packet)

    def ticks(pts: int) -> int:
        # The fps filter's own rounding: av_rescale_q_rnd(NEAR_INF) of the shifted pts.
        value = Fraction(pts - plan_.shift) * info.time_base * plan_.fps
        return int(value + Fraction(1, 2)) if value >= 0 else -int(-value + Fraction(1, 2))

    last = None  # latest pts fed; NUT refuses a timestamp that does not increase
    try:
        for pts, nv12 in decoder.frames_from(plan_.first_pts):
            if plan_.end_pts is not None and pts >= plan_.end_pts:
                break
            # A splice can present one picture behind the clock (Etcheverry-Atmane: 2400.03 s
            # after 2406.02 s); muxing it raised here and ended the feed early with exit 0.
            # As on the CPU path: the fps filter drops it (a repeated pts only replaces the
            # held picture of its tick below), and a CFR output (no fps filter) shows it in
            # the next frame slot.
            if plan_.fps is not None and last is not None and pts < last:
                if OUT_OF_ORDER == "drop":
                    _note(f"order:{plan_.video}", f"dropped a picture out of order at pts {pts}")
                    continue
            elif plan_.fps is None and last is not None and pts <= last and OUT_OF_ORDER == "drop":
                if "passthrough" in plan_.piped:
                    # A passthrough output keeps each picture's own clock, which NUT cannot
                    # carry backwards: the output ends here, recorded as a fallback event.
                    _note(
                        f"order:{plan_.video}",
                        f"picture out of order at pts {pts}; passthrough output ends here",
                        "out_of_order_truncated",
                    )
                    break
                _note(f"order:{plan_.video}", f"re-timed a picture out of order at pts {pts}")
                pts = last + frame_ticks
            if (
                last is None
                and plan_.pad_start
                and START_PAD == "on"
                and round(Fraction(pts - plan_.first_pts, frame_ticks)) >= 1
            ):
                # One broadcaster's files: video starts at 0.033 s, the container (audio) at 0. The CPU
                # path's CFR output repeats the first picture from 0; so does this one now.
                send(plan_.first_pts, nv12)
            last = pts
            if plan_.fps is None:
                send(pts, nv12)
                continue
            current = ticks(pts)
            if held is not None and current != tick:
                send(*held)
            held, tick = (pts, nv12.clone()), current
        if held is not None:
            send(*held)
        muxer.close()
    except (BrokenPipeError, ValueError, OSError):
        pass  # ffmpeg stopped reading (-frames:v reached) or failed; its exit code reports it
    finally:
        try:
            stdin.close()
        except OSError:
            pass


def _nvdec_plan(command: list[str], kwargs: dict):
    """(plan, source, gpu) when the switch is on and NVDEC can stand in exactly, else None."""
    gpu = requested_gpu()
    if gpu is None or "stdin" in kwargs or "input" in kwargs:
        return None
    try:
        piped = plan(command)
        if piped is None:
            _note(
                " ".join(command[:3]),
                f"command form not recognised, CPU decode: {command}",
                "command_not_recognised",
            )
            return None
        import PyNvVideoCodec  # noqa: F401

        info = probe(piped.video)
        if info is None:
            _note(
                f"fmt:{piped.video}",
                f"{piped.video} is not 8-bit 4:2:0; CPU decode",
                "unsupported_source",
            )
            return None
        if (os.path.realpath(piped.video), gpu) not in _decoders:
            import torch

            # PyNvVideoCodec exits the process (no exception) when the card is out of memory,
            # e.g. a production job landed on it: check first, the CPU path gives the same bytes.
            free_mib = torch.cuda.mem_get_info(gpu)[0] >> 20
            if free_mib < MIN_FREE_MIB:
                _note(f"mem:{gpu}", f"cuda:{gpu} has {free_mib} MiB free; CPU decode", "low_memory")
                return None
        _decoder(piped.video, gpu, info)
    except Exception as error:  # no library, or the decoder refused this codec/GPU
        _note(f"open:{command}", f"NVDEC unavailable ({error}); CPU decode", _open_kind(error))
        return None
    return piped, info, gpu


def _start(found, **kwargs) -> subprocess.Popen:
    piped, info, gpu = found
    process = subprocess.Popen(piped.piped, stdin=subprocess.PIPE, **kwargs)
    feed, process.stdin = process.stdin, None  # the feeder owns it; communicate() must not
    try:  # 3 MB pictures through Linux's default 64 KB pipe cost a context switch per 64 KB
        import fcntl

        fcntl.fcntl(feed.fileno(), 1031, 1 << 20)  # F_SETPIPE_SZ
    except OSError:
        pass
    feeder = threading.Thread(target=_feed, args=(piped, info, gpu, feed), daemon=True)
    feeder.start()
    process._nvdec_feeder = feeder  # type: ignore[attr-defined]
    return process


def popen(command: list[str], **kwargs) -> subprocess.Popen:
    """``subprocess.Popen(command, **kwargs)``, decoding the input on NVDEC when switched on."""
    found = _nvdec_plan(command, kwargs)
    if found is None:
        return subprocess.Popen(command, **kwargs)
    return _start(found, **kwargs)


def run(command: list[str], *, check: bool = False, **kwargs) -> subprocess.CompletedProcess:
    """``subprocess.run(command, check=check, **kwargs)``, decoding on NVDEC when switched on."""
    found = _nvdec_plan(command, kwargs)
    if found is None:
        return subprocess.run(command, check=check, **kwargs)
    if kwargs.pop("capture_output", False):
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    process = _start(found, **kwargs)
    stdout, stderr = process.communicate()
    process._nvdec_feeder.join()  # type: ignore[attr-defined]
    if check and process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def decoded_pts_times(video: str) -> tuple[Iterator[float], dict] | None:
    """Every decoded frame's pts in seconds, as ``ffprobe -show_frames`` prints ``pts_time``
    (``%f`` of pts * av_q2d(time base)), decoded on NVDEC; or None when the switch is off or
    NVDEC cannot decode this source. The returned status dict is filled when the iterator is
    exhausted: ``packets``/``frames`` (a decode that drops a frame fails the audit closed,
    as a decoder error does on the CPU), ``error``, ``decoder``."""
    gpu = requested_gpu()
    if gpu is None:
        return None
    try:
        import PyNvVideoCodec as nvc

        info = probe(video)
        if info is None:
            return None
        _sleep_while_waiting(gpu)
        demuxer = nvc.CreateDemuxer(filename=video)
        decoder = nvc.CreateDecoder(
            gpuid=gpu,
            codec=demuxer.GetNvCodecId(),
            cudacontext=0,
            cudastream=0,
            usedevicememory=True,
        )
    except Exception as error:
        _note(
            f"pts:{video}",
            f"NVDEC unavailable for the timestamp audit ({error}); CPU decode",
            _open_kind(error),
        )
        return None
    scale = info.time_base.numerator / info.time_base.denominator
    status = {
        "decoder": f"nvdec:{gpu}",
        "library": f"PyNvVideoCodec {nvc.__version__}",
        "packets": 0,
        "frames": 0,
        "error": "",
    }

    def times() -> Iterator[float]:
        try:
            for packet in demuxer:
                status["packets"] += packet.bsl > 0  # the demuxer ends with an empty packet
                for frame in decoder.Decode(packet):
                    status["frames"] += 1
                    yield float("%f" % (frame.getPTS() * scale))
            for frame in decoder.Decode(nvc.PacketData()):
                status["frames"] += 1
                yield float("%f" % (frame.getPTS() * scale))
        except Exception as error:  # surfaced as a decoder error: the audit fails closed
            status["error"] = f"{type(error).__name__}: {error}"

    return times(), status
