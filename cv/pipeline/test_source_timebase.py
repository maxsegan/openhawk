import math
import shutil
import subprocess

import pytest

from cv.pipeline.source_timebase import audit_source_timebase, audit_timestamps, frame_pts


def test_compact_parser_keeps_missing_frame_pts_and_ignores_side_data():
    assert list(
        frame_pts(
            [
                "frame|pts_time=0.000000|side_data=sei\n",
                "side_data|ignored=true\n",
                "frame|pts_time=N/A\n",
                "frame\n",
                "frame|pts_time=0.120000\n",
            ]
        )
    ) == [0.0, None, None, 0.12]


@pytest.mark.parametrize("fps", [25.0, 50.0, 30000 / 1001, 60000 / 1001])
def test_global_pts_accepts_native_cadence_and_nonzero_origin(fps):
    report = audit_timestamps((round(12.345 + i / fps, 3) for i in range(400)), fps)
    assert report["nominal_timeline_valid"] is True
    assert report["frames"] == 400
    assert report["first_pts_seconds"] == 12.345


@pytest.mark.parametrize("bad", [None, math.nan, math.inf, 4.0, 2.0])
def test_gap_duplicate_backward_or_missing_pts_cannot_hide_between_samples(bad):
    timestamps = [i / 25 for i in range(500)]
    timestamps[100] = bad  # 4.0 duplicates the next row after we move that row below.
    timestamps[101] = 4.0
    report = audit_timestamps(iter(timestamps), 25.0)
    assert report["nominal_timeline_valid"] is False
    assert report["invalid_frames"] >= 1


def test_regular_local_intervals_cannot_hide_global_clock_drift():
    report = audit_timestamps((i / 25 + i * 0.0001 for i in range(20000)), 25.0)
    assert report["nominal_timeline_valid"] is False
    assert report["maximum_nominal_phase_error_seconds"] > 1.0
    assert len(report["invalid_examples"]) == 32


@pytest.mark.parametrize("values", [[], [0.0]])
def test_insufficient_frames_are_not_a_valid_match_clock(values):
    assert audit_timestamps(values, 25.0)["nominal_timeline_valid"] is False


@pytest.mark.parametrize("fps", [0.0, -25.0, math.nan, math.inf])
def test_invalid_cadence_fails(fps):
    with pytest.raises(ValueError, match="finite and positive"):
        audit_timestamps([], fps)


@pytest.mark.skipif(not shutil.which("ffprobe") or not shutil.which("ffmpeg"), reason="ffmpeg")
def test_real_decoded_b_frame_stream_preserves_presentation_order(tmp_path):
    video = tmp_path / "native.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=128x96:rate=30000/1001",
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            str(video),
        ],
        check=True,
    )
    report = audit_source_timebase(video, 30000 / 1001)
    assert report["completed"] is True
    assert report["nominal_timeline_valid"] is True
    assert report["frames"] == 30
    bad = audit_source_timebase(tmp_path / "missing.mp4", 25.0)
    assert bad["completed"] is False
    assert bad["nominal_timeline_valid"] is False


@pytest.mark.skipif(not shutil.which("ffprobe") or not shutil.which("ffmpeg"), reason="ffmpeg")
def test_video_clock_does_not_initialize_unrelated_opus_eof_parser(tmp_path):
    video = tmp_path / "generated_valid_opus.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=128x96:rate=25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-c:a",
            "libopus",
            str(video),
        ],
        check=True,
    )
    report = audit_source_timebase(video, 25.0)
    assert report["frames"] == 25
    assert report["completed"] is True
    assert report["nominal_timeline_valid"] is True
    assert report["decoder_error_excerpt"] == ""
    assert report["stream_information_discovery"] is False
    assert report["audio_integrity"] == "not_audited_by_video_clock_scan"
    assert "-nofind_stream_info" in report["command"]
