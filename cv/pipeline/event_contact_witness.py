"""Bounded source audio/pose contact witnesses; no event truth or learned fusion.

Audio flux reuses the production contact producer. Pose energy reuses the older
witness experiment's normalized wrist/elbow speed, but never interpolates across
missing pictures or cuts. Scores indicate support, not calibrated probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
import subprocess

import numpy as np
from scipy.signal import find_peaks, peak_widths

from cv.pipeline.contacts_audio import onset_scores

SAMPLE_RATE = 16_000
FFT_SAMPLES = 512
HOP_SAMPLES = 160
JOINTS = ("left_elbow", "right_elbow", "left_wrist", "right_wrist")


@dataclass(frozen=True)
class ClockedAudio:
    samples: np.ndarray
    first_pts_seconds: float
    sample_rate: int
    receipt: dict


def validate_audio_frames(frames: list[tuple[int, float, int]], samples: int, rate: int) -> dict:
    """Bind decoded samples to actual post-resampling timestamps, not seek intent."""
    if not frames or rate <= 0:
        raise ValueError("decoded audio PTS and positive cadence required")
    first = frames[0][1]
    seen = 0
    maximum_error = 0.0
    for expected, (index, pts, count) in enumerate(frames):
        if index != expected or count < 1 or not math.isfinite(pts):
            raise ValueError("invalid decoded audio frame identity")
        error = abs(pts - first - seen / rate)
        maximum_error = max(maximum_error, error)
        # ashowinfo prints decimal seconds with limited precision; sample PTS is
        # separately present in its records and must remain exact at the caller.
        if error > max(0.0011, 2 / rate):
            raise ValueError("decoded audio has a gap/discontinuity")
        seen += count
    if seen != samples:
        raise ValueError("audio timestamps do not cover every emitted sample")
    return {
        "first_pts_seconds": first,
        "samples": seen,
        "sample_rate": rate,
        "maximum_continuity_error_seconds": maximum_error,
        "source_timestamp_shift_seconds": 0.0,
        "clock": "decoded_source_pts",
    }


def decode_clocked_audio(video: Path, start: float, end: float) -> ClockedAudio:
    """Decode a bounded source-time interval and validate actual output A/V clock.

    copyts retains the source epoch; atrim names absolute PTS. No asetpts, frame
    rate reconstruction, broadcast-offset fitting or historical reel shift.
    """
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        raise ValueError("finite ordered source audio interval required")
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "info",
        "-copyts",
        "-ss",
        str(max(0, start - 1)),
        "-t",
        str(end - start + 2),
        "-i",
        str(video),
        "-map",
        "0:a:0",
        "-vn",
        "-af",
        f"atrim=start={start}:end={end},aresample={SAMPLE_RATE},ashowinfo",
        "-ac",
        "1",
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=True)
    samples = np.frombuffer(result.stdout, dtype="<f4").copy()
    records = []
    for line in result.stderr.decode(errors="replace").splitlines():
        if "ashowinfo" not in line or "nb_samples:" not in line:
            continue
        fields = dict(re.findall(r"\b(n|pts|pts_time|rate|nb_samples):([^\s]+)", line))
        if int(fields["rate"]) != SAMPLE_RATE:
            raise ValueError("unexpected decoded audio cadence")
        # Integer output PTS use the declared post-resampling 1/rate timebase.
        records.append(
            (int(fields["n"]), int(fields["pts"]) / SAMPLE_RATE, int(fields["nb_samples"]))
        )
    receipt = validate_audio_frames(records, len(samples), SAMPLE_RATE)
    if not start - 2 / SAMPLE_RATE <= receipt["first_pts_seconds"] <= start + 0.02:
        raise ValueError("decoded audio source epoch does not match requested window")
    return ClockedAudio(
        samples,
        receipt["first_pts_seconds"],
        SAMPLE_RATE,
        receipt | {"command": command, "requested_source_interval": [start, end]},
    )


def audio_peaks(audio: ClockedAudio) -> list[dict]:
    scores = onset_scores(audio.samples, audio.sample_rate, FFT_SAMPLES, HOP_SAMPLES)
    indices, properties = find_peaks(scores, height=5.0, prominence=2.5, distance=3)
    widths = peak_widths(scores, indices)[0] if len(indices) else []
    return [
        {
            "source_pts_seconds": audio.first_pts_seconds
            + (float(index) * HOP_SAMPLES + FFT_SAMPLES / 2) / audio.sample_rate,
            "score_z": float(height),
            "prominence_z": float(prominence),
            "width_seconds": float(width * HOP_SAMPLES / audio.sample_rate),
            "support": min(1.0, float(prominence) / 5.0),
            "kind": "untyped_audio_transient",
        }
        for index, height, prominence, width in zip(
            indices, properties["peak_heights"], properties["prominences"], widths, strict=True
        )
        if width * HOP_SAMPLES / audio.sample_rate <= 0.06
    ]


def audio_support(peaks: list[dict], source_interval: tuple[float, float]) -> dict:
    """Occurrence support only; never infer an exact contact epoch from sound.

    A fixed 0..120ms acoustic-arrival window accommodates sound travel to a
    broadcast microphone. It is not a fitted/assumed A/V offset. Multiple
    transients abstain rather than choosing the one most favorable to S6.
    """
    low, high = source_interval
    eligible = [row for row in peaks if low <= row["source_pts_seconds"] <= high + 0.12]
    if len(eligible) != 1:
        return {
            "available": True,
            "supported": False,
            "reason": "no_unique_audio_transient",
            "transients": len(eligible),
        }
    return {
        "available": True,
        "supported": True,
        "support": eligible[0]["support"],
        "transient": eligible[0],
        "event_time_refined": False,
        "acoustic_arrival_allowance_seconds": [0.0, 0.12],
    }


def pose_motion(rows: list[dict], cuts: set[int]) -> list[dict]:
    """Three genuinely observed neighboring epochs; no interpolation or cross-cut smoothing.

    Caller supplies native pixel boxes/joints, native PTS, and one stable tracked
    player identity per series. The required schema is deliberately explicit.
    """
    ordered = sorted(rows, key=lambda row: row["frame"])
    if len({row["frame"] for row in ordered}) != len(ordered):
        raise ValueError("one pose row per native frame and tracked player required")
    output = []
    for i, center in enumerate(ordered):
        result = {
            "frame": center["frame"],
            "source_pts_seconds": center["source_pts_seconds"],
            "supported": False,
            "reason": "missing_observed_derivative_support",
        }
        output.append(result)
        if i == 0 or i == len(ordered) - 1:
            continue
        local = ordered[i - 1 : i + 2]
        if (
            [int(x["frame"]) for x in local]
            != list(range(int(center["frame"]) - 1, int(center["frame"]) + 2))
            or any(int(x["frame"]) in cuts for x in local[1:])
            or len({x["track_id"] for x in local}) != 1
        ):
            continue
        times = np.array([x["source_pts_seconds"] for x in local], float)
        if (
            not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
            or max(np.diff(times)) > 0.08
        ):
            continue
        speeds = []
        for joint in JOINTS:
            values = []
            for row in local:
                point = row.get("joints", {}).get(joint)
                box = row.get("box_xyxy_native")
                if (
                    point is None
                    or box is None
                    or float(point[2]) < 0.15
                    or not np.isfinite([*point, *box]).all()
                    or box[3] <= box[1]
                ):
                    break
                values.append((np.asarray(point[:2]) - np.asarray(box[:2])) / (box[3] - box[1]))
            if len(values) == 3:
                velocity = (values[2] - values[0]) / (times[2] - times[0])
                speeds.append(float(np.linalg.norm(velocity)) * (0.5 if "elbow" in joint else 1))
        if len(speeds) < 2:
            continue
        result.update(
            available=True,
            reason=None,
            energy=max(speeds),
            supported=True,
            observed_frames=[int(x["frame"]) for x in local],
        )
    values = np.array([row["energy"] for row in output if row["supported"]], float)
    if len(values):
        median = float(np.median(values))
        mad = max(1.4826 * float(np.median(abs(values - median))), 1e-4)
        for row in output:
            if row["supported"]:
                row["score_z"] = (row["energy"] - median) / mad
    return output
