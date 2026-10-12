import hashlib
import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from cv.pipeline import nvdec_video, players

INFO = nvdec_video.SourceInfo(
    width=1920,
    height=1080,
    time_base=Fraction(1, 1000),
    start_pts=0,
    rate=Fraction(25),
    sar=Fraction(1),
    color={},
    tags="range=tv:colorspace=bt709",
)


def _plan(tmp_path, command):
    video = tmp_path / "v.mkv"
    video.write_bytes(b"")
    return nvdec_video.plan([c.replace("VIDEO", str(video)) for c in command], lambda _: INFO)


def test_whole_video_fps_chain_keeps_the_commands_filters(tmp_path):
    plan = _plan(
        tmp_path,
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            "VIDEO",
            "-vf",
            "fps=1,scale=960:-2",
            "-q:v",
            "4",
            "o/s_%06d.jpg",
        ],
    )
    assert plan.piped[:9] == [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-copyts",
        "-f",
        "nut",
        "-i",
        "-",
    ]
    assert plan.piped[9:] == [
        "-vf",
        "settb=1/1000,setsar=1/1,setparams=range=tv:colorspace=bt709,fps=1,scale=960:-2",
        "-q:v",
        "4",
        "o/s_%06d.jpg",
    ]
    assert (plan.first_pts, plan.shift, plan.end_pts, plan.fps) == (0, 0, None, Fraction(1))


def test_accurate_seek_span_shifts_like_ffmpeg(tmp_path):
    plan = _plan(
        tmp_path,
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            "36.5",
            "-i",
            "VIDEO",
            "-t",
            "7.5",
            "-map",
            "0:v:0",
            "-f",
            "rawvideo",
            "-",
        ],
    )
    assert (plan.first_pts, plan.shift, plan.end_pts) == (36500, 36500, 44000)
    assert "-t" not in plan.piped


def test_native_point_extraction_keeps_timestamps(tmp_path):
    command, _, _ = players.native_extract_command(
        str(tmp_path / "v.mkv"),
        "f_%04d.jpg",
        250,
        349,
        25.0,
        native=True,
        measured_native_epochs=True,
    )
    (tmp_path / "v.mkv").write_bytes(b"")
    plan = nvdec_video.plan(command, lambda _: INFO)
    assert (plan.first_pts, plan.shift) == (9000, 0)
    assert "select=gte(pts\\,floor(10/TB+0.5))" in plan.piped[plan.piped.index("-vf") + 1]


@pytest.mark.parametrize(
    "command",
    [
        ["ffmpeg", "-ss", "1", "-t", "2", "-i", "VIDEO", "-f", "null", "-"],  # input -t
        ["ffmpeg", "-i", "VIDEO", "-i", "VIDEO", "-f", "null", "-"],
        ["ffprobe", "-i", "VIDEO"],
        ["ffmpeg", "-copyts", "-i", "VIDEO", "-f", "null", "-"],  # copyts with an accurate seek
    ],
)
def test_unrecognised_forms_fall_back(tmp_path, command):
    assert _plan(tmp_path, command) is None


def test_switch_off_is_plain_subprocess(monkeypatch):
    monkeypatch.delenv("TENNIS_VIDEO_DECODE", raising=False)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    nvdec_video.run(["ffmpeg", "-i", "x"], check=True, stderr=None)
    assert calls == [(["ffmpeg", "-i", "x"], {"check": True, "stderr": None})]


def test_full_card_falls_back_before_opening(monkeypatch, tmp_path):
    # PyNvVideoCodec exits the process on CUDA_ERROR_OUT_OF_MEMORY, so a short card must
    # never reach the decoder.
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("TENNIS_VIDEO_DECODE", "nvdec:0")
    monkeypatch.setitem(__import__("sys").modules, "PyNvVideoCodec", object())
    monkeypatch.setattr(nvdec_video, "probe", lambda _: INFO)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (512 << 20, 96 << 30))
    opened = []
    monkeypatch.setattr(nvdec_video, "_decoder", lambda *a: opened.append(a))
    video = tmp_path / "v.mkv"
    video.write_bytes(b"")
    command = ["ffmpeg", "-i", str(video), "-f", "null", "-"]
    assert nvdec_video.plan(command, lambda _: INFO) is not None
    assert nvdec_video._nvdec_plan(command, {}) is None
    assert opened == []


def _gpu_ready() -> bool:
    try:
        import PyNvVideoCodec  # noqa: F401
        import torch

        return torch.cuda.is_available() and shutil.which("ffmpeg") is not None
    except ImportError:
        return False


@pytest.mark.skipif(not _gpu_ready(), reason="needs PyNvVideoCodec, a CUDA GPU and ffmpeg")
def test_nvdec_outputs_are_byte_identical(tmp_path, monkeypatch):
    # Seek between the extraction windows (forward, ahead, then behind) on this short clip.
    monkeypatch.setattr(nvdec_video, "SEEK_AHEAD_SECONDS", 0.5)
    gpu = os.environ.get("TENNIS_NVDEC_TEST_GPU", "0")
    seeks = []
    real_open = nvdec_video._Decoder._open

    def recording_open(self, seek_index=None):
        seeks.append(seek_index)
        real_open(self, seek_index)

    monkeypatch.setattr(nvdec_video._Decoder, "_open", recording_open)
    video = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1920x1080:rate=25:duration=4",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-g",
            "30",
            str(video),
        ],
        check=True,
    )
    base = ["ffmpeg", "-nostdin", "-v", "error"]
    commands = {
        "overlay": lambda d: [
            *base,
            "-i",
            str(video),
            "-vf",
            "fps=1,scale=960:-2",
            "-q:v",
            "4",
            str(d / "s_%06d.jpg"),
        ],
        "shot": lambda d: [
            *base,
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-vf",
            "scale=96:54,format=gray",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
    }
    for first, last in ((40, 69), (85, 95), (10, 20)):
        commands[f"extract_{first}"] = lambda d, a=first, b=last: players.native_extract_command(
            str(video), str(d / "f_%04d.jpg"), a, b, 25.0, native=True, measured_native_epochs=True
        )[0]
    for name, make in commands.items():
        digests = []
        for switch in ("cpu", f"nvdec:{gpu}"):
            monkeypatch.setenv("TENNIS_VIDEO_DECODE", switch)
            d = tmp_path / name / switch
            d.mkdir(parents=True)
            with (d / "log").open("w") as log:
                done = nvdec_video.run(make(d), check=True, stdout=subprocess.PIPE, stderr=log)
            files = sorted(d.glob("*.jpg"))
            if name.startswith("extract"):
                clock = players.captured_native_clock(d / "log", [p.name for p in files], 0.6)
                files.append(Path(str(clock)))
            digests.append(
                [
                    hashlib.sha256(done.stdout or b"").hexdigest(),
                    *[
                        hashlib.sha256(
                            p.read_bytes() if p.exists() else str(p).encode()
                        ).hexdigest()
                        for p in files
                    ],
                ]
            )
        assert digests[0] == digests[1], name
        assert len(digests[0]) > 1 or name == "shot"
    assert len([index for index in seeks if index]) >= 2, seeks  # windows at 40 and 85


@pytest.mark.skipif(not _gpu_ready(), reason="needs PyNvVideoCodec, a CUDA GPU and ffmpeg")
@pytest.mark.parametrize("policy", ["drop", "stop"])
def test_a_picture_out_of_order_does_not_end_the_feed(tmp_path, monkeypatch, policy):
    """A splice re-presents a picture at a timestamp already fed (here pts 5.00 s twice): the
    NUT muxer refused it and the feed ended early with exit 0. ``drop`` matches the CPU."""
    gpu = os.environ.get("TENNIS_NVDEC_TEST_GPU", "0")
    clean, video = tmp_path / "clean.mp4", tmp_path / "splice.mkv"
    lavfi = "testsrc2=size=1920x1080:rate=25:duration=4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", lavfi, "-c:v", "libx264"]
        + ["-bf", "0", "-g", "30", "-pix_fmt", "yuv420p", str(clean)],
        check=True,
    )
    jump = "setts=ts='if(gte(N\\,50)*not(eq(N\\,51))\\,TS+3/TB\\,TS)'"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(clean), "-c", "copy", "-bsf:v", jump]
        + [str(video)],
        check=True,
    )
    monkeypatch.setattr(nvdec_video, "OUT_OF_ORDER", policy)
    base = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(video)]
    shot = [*base, "-map", "0:v:0", "-vf", "scale=96:54,format=gray"]
    shot += ["-f", "rawvideo", "-pix_fmt", "gray", "-"]
    outputs = {}
    for switch in ("cpu", f"nvdec:{gpu}"):
        monkeypatch.setenv("TENNIS_VIDEO_DECODE", switch)
        stream = nvdec_video.run(shot, check=True, capture_output=True).stdout
        d = tmp_path / switch.replace(":", "_")
        d.mkdir()
        nvdec_video.run([*base, "-vf", "fps=10,scale=960:-2", str(d / "s_%04d.jpg")], check=True)
        outputs[switch] = (stream, [p.read_bytes() for p in sorted(d.glob("*.jpg"))])
    cpu, nvdec = outputs.values()
    assert len(cpu[0]) == 175 * 96 * 54 and len(cpu[1]) == 70
    assert nvdec[1] == cpu[1]  # the fps chain keeps the later of two equal timestamps
    if policy == "drop":
        assert nvdec[0] == cpu[0]
    else:  # rollback: the stream ends at the repeated timestamp
        assert len(nvdec[0]) < len(cpu[0])


@pytest.mark.skipif(not _gpu_ready(), reason="needs PyNvVideoCodec, a CUDA GPU and ffmpeg")
@pytest.mark.parametrize("pad", ["on", "off"])
def test_a_video_starting_after_its_audio_keeps_cpu_frame_indices(tmp_path, monkeypatch, pad):
    """One broadcaster's files: video starts at 0.033 s, audio (the container start) at 0. The CPU CFR
    stream repeats the first picture from 0, so frame index / fps stays source time."""
    gpu = os.environ.get("TENNIS_NVDEC_TEST_GPU", "0")
    video = tmp_path / "late_video.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "sine=duration=3"]
        + ["-itsoffset", "0.08", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=25:duration=2"]
        + ["-map", "1:v", "-map", "0:a", "-c:v", "libx264", "-bf", "0", "-pix_fmt", "yuv420p"]
        + ["-c:a", "aac", str(video)],
        check=True,
    )
    monkeypatch.setattr(nvdec_video, "START_PAD", pad)
    stream = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(video), "-map", "0:v:0", "-vf"]
    stream += ["scale=96:54,format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    outputs = []
    for switch in ("cpu", f"nvdec:{gpu}"):
        monkeypatch.setenv("TENNIS_VIDEO_DECODE", switch)
        outputs.append(nvdec_video.run(stream, check=True, capture_output=True).stdout)
    frame = 96 * 54
    assert outputs[0][:frame] == outputs[0][frame : 2 * frame]  # CPU repeats the first picture
    if pad == "on":
        assert outputs[1] == outputs[0]
    else:  # rollback: indices start at the first picture, two slots early
        assert outputs[1] == outputs[0][2 * frame :]
