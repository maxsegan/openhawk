"""Native-resolution crop, context and audio dataset for the S5 event video model.

For every label-free candidate window produced by
:func:`cv.pipeline.event_model_v2_features.build_automatic_dataset` this module cuts

* a 16-frame ``192x108`` native crop sequence centred on the tracked ball,
  sampled from unique exposures only (no resizing of the source, zero padding at
  the frame edge),
* one ``384x216`` native context crop on the centre frame, for court lines, and
* a 0.5 s log-mel spectrogram from the 16 kHz broadcast audio,

and writes them as one uncompressed ``.npz`` shard per broadcast with a manifest
that records broadcast, clip, centre frame and crop geometry.  Truth is joined
only afterwards, by :func:`join_truth`, so the cut is label-free.
"""

from __future__ import annotations

import argparse
import ast
import bisect
import csv
import json
import math
import struct
import subprocess
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cv.pipeline.artifact_cache import (
    receipt_path,
    stage_identity,
    stage_receipt_matches,
    write_stage_receipt,
)
from cv.pipeline.automatic_ball_track import (
    CURRENT_EVENT_TRACK,
    LEGACY_EVENT_TRACK,
    current_track_inputs,
    validate_event_track,
)
from cv.pipeline.event_model_v2_features import (
    RADIUS,
    TRACK_NAME,
    FRAME_HOMOGRAPHY_NAME,
    COURT_GEOMETRY_MODES,
    COURT_FRAME_MISSING_POLICIES,
    bind_broadcast_track,
    load_court_transport,
    validate_court_transport,
    build_automatic_dataset,
)

SCHEMA = "event_crops_v1"
ALTERNATE_SCHEMA = "event_crops_alternate_proposals_v1"
FRAMES_DIR = "audit_frames_native_1080"
REEL_NAME = "audit_reel_native_1080.mp4"
AUDIO_CACHE_NAME = "contact_audio_scores_16k_native_v1.npz"
CADENCE_NAME = "frame_cadence_v1.json"

NATIVE_WIDTH = 1920
NATIVE_HEIGHT = 1080
TIGHT_WIDTH = 192
TIGHT_HEIGHT = 108
CONTEXT_WIDTH = 384
CONTEXT_HEIGHT = 216
SEQUENCE_LENGTH = 16
SEQUENCE_RADIUS = 8

SAMPLE_RATE = 16_000
AUDIO_SECONDS = 0.5
MEL_BANDS = 64
MEL_HOP = 160
MEL_FFT = 512
MEL_FRAMES = 50
MEL_FLOOR = 1e-8

TRUTH_ANCHOR_TOLERANCE = 0.500001


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def crop_bounds(centre_x: float, centre_y: float, width: int, height: int) -> tuple[int, int]:
    """Return the top-left native pixel of a ``width x height`` crop.

    The crop is centred on ``(centre_x, centre_y)`` and clamped so that it stays
    inside the native frame whenever the frame is large enough; the caller pads
    whatever still falls outside.
    """

    left = int(round(centre_x)) - width // 2
    top = int(round(centre_y)) - height // 2
    left = max(0, min(left, NATIVE_WIDTH - width))
    top = max(0, min(top, NATIVE_HEIGHT - height))
    return left, top


def cut(frame: np.ndarray, left: int, top: int, width: int, height: int) -> np.ndarray:
    """Cut a native crop with zero padding for anything outside the frame."""

    output = np.zeros((height, width, 3), dtype=np.uint8)
    if frame is None:
        return output
    source_left = max(0, left)
    source_top = max(0, top)
    source_right = min(frame.shape[1], left + width)
    source_bottom = min(frame.shape[0], top + height)
    if source_right <= source_left or source_bottom <= source_top:
        return output
    output[source_top - top : source_bottom - top, source_left - left : source_right - left] = (
        frame[source_top:source_bottom, source_left:source_right]
    )
    return output


# --------------------------------------------------------------------------- #
# unique exposures
# --------------------------------------------------------------------------- #
def unique_exposures(duplicate_frames: list[int], n_frames: int) -> list[int]:
    """Return the frame indices that carry a distinct exposure.

    ``duplicate_frames`` is the cadence audit's list of frames that repeat the
    previous exposure; those frames are dropped, so consecutive entries of the
    result are always different pictures.
    """

    dropped = {int(value) for value in duplicate_frames}
    return [frame for frame in range(1, n_frames + 1) if frame not in dropped]


def sequence_frames(exposures: list[int], centre: int) -> list[int]:
    """Return ``SEQUENCE_LENGTH`` unique-exposure frames around ``centre``.

    The centre frame keeps position ``SEQUENCE_RADIUS`` in the returned list; the
    sequence is clamped to the available exposures and, at a clip boundary,
    repeats the nearest available exposure so the tensor shape is constant.
    """

    if not exposures:
        return [centre] * SEQUENCE_LENGTH
    position = bisect.bisect_left(exposures, centre)
    position = min(position, len(exposures) - 1)
    output = []
    for offset in range(-SEQUENCE_RADIUS, SEQUENCE_LENGTH - SEQUENCE_RADIUS):
        index = min(max(position + offset, 0), len(exposures) - 1)
        output.append(exposures[index])
    return output


# --------------------------------------------------------------------------- #
# tracks
# --------------------------------------------------------------------------- #
NATIVE_COLUMNS_KEY = "native_1920x1080"
LEGACY_COLUMNS_KEY = "legacy_960x540"


def track_coordinate_space(sidecar: Path, columns: list[str]) -> dict:
    """Return which track columns to read and the scale that makes them native.

    A track sidecar records the size the artifact's coordinates are written in.
    Older tracks write one pair, ``x``/``y``, in ``artifact_size``.  Newer ones
    keep those legacy numbers under the same two names, add native
    ``x_native``/``y_native``, and declare the artifact itself native -- so
    reading ``x``/``y`` and trusting ``artifact_size`` puts every crop at half
    the true position with no error anywhere.  The declared column map therefore
    wins over the bare sizes whenever the sidecar carries one.
    """

    manifest = json.loads(sidecar.read_text())
    image = manifest["image_size"]
    declared = manifest.get("coordinate_columns") or {}
    native = declared.get(NATIVE_COLUMNS_KEY)
    if native:
        if not all(name in columns for name in native):
            raise ValueError(
                f"{sidecar} declares native columns {native} that the track does not carry"
            )
        return {
            "space": NATIVE_COLUMNS_KEY,
            "x_column": native[0],
            "y_column": native[1],
            "scale_x": float(image["width"]) / NATIVE_WIDTH,
            "scale_y": float(image["height"]) / NATIVE_HEIGHT,
        }
    legacy = declared.get(LEGACY_COLUMNS_KEY)
    size = manifest.get("legacy_artifact_size") if legacy else None
    size = size or manifest["artifact_size"]
    pair = legacy if legacy and all(name in columns for name in legacy) else ["x", "y"]
    return {
        "space": LEGACY_COLUMNS_KEY if legacy else "artifact_size",
        "x_column": pair[0],
        "y_column": pair[1],
        "scale_x": float(image["width"]) / float(size["width"]),
        "scale_y": float(image["height"]) / float(size["height"]),
    }


def read_track(
    path: Path, x_column: str = "x", y_column: str = "y"
) -> dict[str, dict[int, tuple[float, float]]]:
    """Read the joint availability track, best score per (clip, frame)."""

    output: dict[str, dict[int, tuple[float, float]]] = {}
    best: dict[tuple[str, int], float] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for name in (x_column, y_column):
            if reader.fieldnames is not None and name not in reader.fieldnames:
                raise ValueError(f"{path} has no column {name}: {reader.fieldnames}")
        for row in reader:
            clip = row["clip"]
            frame = int(Path(row["frame"]).stem.removeprefix("f_"))
            score = float(row.get("score") or 0.0)
            key = (clip, frame)
            if key in best and best[key] >= score:
                continue
            best[key] = score
            output.setdefault(clip, {})[frame] = (
                float(row[x_column]),
                float(row[y_column]),
            )
    return output


def track_centres(
    observations: dict[int, tuple[float, float]],
    frames: list[int],
    scale_x: float,
    scale_y: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return native crop centres for ``frames`` by interpolating the track.

    Gaps are filled by linear interpolation between the neighbouring detections
    and held constant outside the detected span, so a crop exists on every frame.
    Frames with no detection of their own are reported through the third value,
    which is the regression supervision mask.
    """

    if not observations:
        centre_x = np.full(len(frames), NATIVE_WIDTH / 2.0, dtype=np.float32)
        centre_y = np.full(len(frames), NATIVE_HEIGHT / 2.0, dtype=np.float32)
        return centre_x, centre_y, np.zeros(len(frames), dtype=bool)
    known = sorted(observations)
    xs = np.asarray([observations[frame][0] * scale_x for frame in known], dtype=np.float64)
    ys = np.asarray([observations[frame][1] * scale_y for frame in known], dtype=np.float64)
    grid = np.asarray(frames, dtype=np.float64)
    centre_x = np.interp(grid, known, xs).astype(np.float32)
    centre_y = np.interp(grid, known, ys).astype(np.float32)
    observed = np.asarray([frame in observations for frame in frames], dtype=bool)
    return centre_x, centre_y, observed


# --------------------------------------------------------------------------- #
# audio
# --------------------------------------------------------------------------- #
def read_audio(video: Path, start: float, duration: float) -> np.ndarray:
    """Decode one mono 16 kHz span of the broadcast reel."""

    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-ss",
        f"{max(start, 0.0):.6f}",
        "-i",
        str(video),
        "-t",
        f"{max(duration, 0.0):.6f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def mel_filterbank(
    bands: int = MEL_BANDS, fft: int = MEL_FFT, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """Return a ``bands x (fft//2+1)`` triangular mel filterbank."""

    def to_mel(hertz: np.ndarray | float) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + np.asarray(hertz, dtype=np.float64) / 700.0)

    def to_hertz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    edges = to_hertz(np.linspace(to_mel(0.0), to_mel(sample_rate / 2.0), bands + 2))
    frequencies = np.fft.rfftfreq(fft, 1.0 / sample_rate)
    filters = np.zeros((bands, len(frequencies)), dtype=np.float32)
    for band in range(bands):
        low, centre, high = edges[band], edges[band + 1], edges[band + 2]
        rising = (frequencies - low) / max(centre - low, 1e-9)
        falling = (high - frequencies) / max(high - centre, 1e-9)
        filters[band] = np.clip(np.minimum(rising, falling), 0.0, None)
    return filters


_FILTERBANK = mel_filterbank()


def log_mel(samples: np.ndarray) -> np.ndarray:
    """Return a ``MEL_BANDS x MEL_FRAMES`` log-mel spectrogram of ``samples``."""

    output = np.zeros((MEL_BANDS, MEL_FRAMES), dtype=np.float32)
    if samples.size < MEL_FFT:
        samples = np.pad(samples, (0, MEL_FFT - samples.size))
    window = np.hanning(MEL_FFT).astype(np.float32)
    starts = np.arange(MEL_FRAMES) * MEL_HOP
    usable = starts[starts + MEL_FFT <= samples.size]
    if not usable.size:
        return output
    index = usable[:, None] + np.arange(MEL_FFT)[None, :]
    spectra = np.abs(np.fft.rfft(samples[index] * window[None, :], axis=1)) ** 2
    energies = spectra @ _FILTERBANK.T
    output[:, : len(usable)] = np.log(energies.T + MEL_FLOOR).astype(np.float32)
    return output


def point_windows(match_root: Path) -> dict[str, tuple[float, float]]:
    """Return the reel time span of every point of a broadcast."""

    cache = match_root / AUDIO_CACHE_NAME
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
        windows = metadata.get("windows") or {}
        if windows:
            return {clip: (float(span[0]), float(span[1])) for clip, span in windows.items()}
    output: dict[str, tuple[float, float]] = {}
    point_map = match_root / "audit_reel_point_map.csv"
    if point_map.exists():
        with point_map.open(newline="") as handle:
            for row in csv.DictReader(handle):
                output[f"pt{int(row['pt']):04d}"] = (
                    float(row["rally_t_start"]),
                    float(row["rally_t_end"]),
                )
    return output


# --------------------------------------------------------------------------- #
# alternate proposal centres
# --------------------------------------------------------------------------- #
ARC_TRACK_NAME = "ball_track_joint_native1080_arc_augmented_v2.csv"
FAR_NATIVE_CANDIDATES = (
    "ball_candidates_tracknetv2_native1080_far_native_v1.csv",
    "ball_candidates_wasb_native1080_far_native_v1.csv",
)
SLIDING_CANDIDATES = (
    "ball_candidates_tracknetv2_native1080_sliding_k5_v1.csv",
    "ball_candidates_wasb_native1080_sliding_k5_v1.csv",
)
ALTERNATE_SOURCES = (
    "arc_track",
    "track_corner",
    "far_native",
    "sliding",
    "physics_departure",
    "pose_swing",
    "bounce_implication",
)
ALTERNATE_MIN_OFFSET_PX = 24.0
"""An alternate centre must move the crop by more than this to be worth cutting.

Below it the alternate sees the same picture as the row the track already
proposed, so the classifier would score a duplicate.
"""

ALTERNATE_SEPARATION_PX = 24.0
"""Two alternates on one frame closer than this collapse to the first source."""

CORNER_MIN_DEG = 45.0
"""``contact_frame_refiner.TURN_GATE_DEG``: below it the track has no corner."""


def _read_native_rows(path: Path) -> dict[str, dict[int, list[tuple[float, float, float]]]]:
    """Every row of a tracker artifact as native pixels, keyed by clip and frame.

    The coordinate sidecar decides the space, exactly as :func:`read_track` does
    for the composed track: a declared native column pair wins, otherwise the
    declared legacy pair is scaled by the sidecar's own sizes.
    """

    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        columns = csv.DictReader(handle).fieldnames or []
    space = track_coordinate_space(Path(f"{path}.coordinates.json"), list(columns))
    x_column, y_column = space["x_column"], space["y_column"]
    scale_x, scale_y = float(space["scale_x"]), float(space["scale_y"])
    output: dict[str, dict[int, list[tuple[float, float, float]]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(Path(row["frame"]).stem.removeprefix("f_"))
            output.setdefault(row["clip"], {}).setdefault(frame, []).append(
                (
                    float(row[x_column]) * scale_x,
                    float(row[y_column]) * scale_y,
                    float(row.get("score") or 0.0),
                )
            )
    return output


def _best_candidate(
    values: list[tuple[float, float, float]] | None,
) -> tuple[float, float] | None:
    if not values:
        return None
    x, y, _score = max(values, key=lambda value: value[2])
    return x, y


def _turn_degrees(track: dict[int, tuple[float, float]], frame: int) -> float:
    """The composed track's direction change at ``frame``, in degrees.

    This is ``contact_frame_refiner.direction_change`` read on one image, which
    is the same corner the contact-anchor refiner votes with.
    """

    from cv.pipeline.contact_frame_refiner import direction_change

    signal = direction_change(track, frame, window=(0,))
    if signal.offset is None:
        return math.nan
    return float(signal.evidence["max_turn_deg"])


def alternate_centres(
    match_root: Path,
    clip: str,
    frames: list[int],
    centre_x: np.ndarray,
    centre_y: np.ndarray,
    observed: np.ndarray,
    *,
    arc_track: dict[int, tuple[float, float]] | None = None,
    far_native: dict[int, list[tuple[float, float, float]]] | None = None,
    sliding: dict[int, list[tuple[float, float, float]]] | None = None,
) -> list[ShardRow]:
    """Return extra crop centres for the frames the composed track cannot place.

    Three label-free sources, in priority order:

    ``arc_track``
        the arc-augmented composed track's own position, wherever the
        availability track has no row of its own on that frame.  Where a
        broadcast carries no availability track at all this is the only centre
        there is, and without it the whole broadcast leaves the event stage
        empty.
    ``far_native``
        the best far-court native detector candidate on the same frames.  It is
        the one witness that does not go through the composed track, so it is
        the only proposal available where the track never locked on.
    ``sliding``
        the best sliding-window native candidate, used only on frames where
        neither of the two above has anything to say.  It is the last witness
        there is, and on a broadcast whose composed track came out empty it is
        the only one.

    An arc-augmented proposal is reported as ``track_corner`` instead of
    ``arc_track`` when the track turns there by at least :data:`CORNER_MIN_DEG`
    -- ``contact_frame_refiner``'s own corner -- so the corner arm can be scored
    on its own.  It is a label on the same pixel, not a separate proposal: the
    corner never names a centre the arc-augmented track has not already named.

    A source is only emitted when it moves the crop by more than
    :data:`ALTERNATE_MIN_OFFSET_PX`; otherwise the classifier would see the same
    picture twice.
    """

    arc_track = arc_track or {}
    far_native = far_native or {}
    sliding = sliding or {}
    rows: list[ShardRow] = []
    for position, frame in enumerate(frames):
        if bool(observed[position]):
            continue
        base = (float(centre_x[position]), float(centre_y[position]))
        proposals: list[tuple[str, tuple[float, float]]] = []
        arc = arc_track.get(frame)
        if arc is not None:
            corner = _turn_degrees(arc_track, frame)
            name = "track_corner" if corner >= CORNER_MIN_DEG else "arc_track"
            proposals.append((name, arc))
        detector = _best_candidate(far_native.get(frame))
        if detector is not None:
            proposals.append(("far_native", detector))
        if not proposals:
            fallback = _best_candidate(sliding.get(frame))
            if fallback is not None:
                proposals.append(("sliding", fallback))
        kept: list[tuple[str, tuple[float, float]]] = []
        for source, point in proposals:
            if math.hypot(point[0] - base[0], point[1] - base[1]) <= ALTERNATE_MIN_OFFSET_PX:
                continue
            if any(
                math.hypot(point[0] - other[0], point[1] - other[1]) <= ALTERNATE_SEPARATION_PX
                for _name, other in kept
            ):
                continue
            kept.append((source, point))
        for source, point in kept:
            rows.append(
                ShardRow(
                    clip=clip,
                    frame=int(frame),
                    centre_x=float(point[0]),
                    centre_y=float(point[1]),
                    track_observed=False,
                    source=source,
                )
            )
    return rows


def scoped_frames(
    root: Path, broadcast: str, *, live_shot_camera: bool = False
) -> dict[str, list[int]]:
    """The active-play frames of every point of a broadcast.

    This repeats ``event_model_v2_features._scope_frames`` on purpose: the
    feature builder skips a broadcast that has no availability track, and the
    proposals below exist precisely to cover that case.
    """

    active = json.loads((root / "active_play_v1.json").read_text())
    output: dict[str, list[int]] = {}
    for point_key, record in sorted(active.items()):
        name, clip = point_key.split("/", 1)
        if name != broadcast:
            continue
        n_frames = int(record["n_frames"])
        spans = record.get("event_spans", [])
        if live_shot_camera:
            from cv.pipeline.camera_frame_support import match_rows
            from cv.pipeline.live_shot_camera import scope_event_spans

            spans = scope_event_spans(
                record, match_rows(root / broadcast, broadcast, [clip])[0]["supported_spans"]
            )
        frames: set[int] = set()
        for start, stop in spans:
            first = max(0, int(math.ceil(float(start))))
            last = min(n_frames - 1, int(math.floor(float(stop))))
            frames.update(range(first, last + 1))
        if frames:
            output[clip] = sorted(frames)
    return output


def _court_normalised(
    homographies: dict[int, np.ndarray], point: int, x: float, y: float
) -> tuple[float | None, float | None]:
    """Project a native pixel onto the court plane, normalised as the features are."""

    from cv.pipeline.event_model_v2_features import COURT_LENGTH_M, COURT_WIDTH_M

    homography = homographies.get(point)
    if homography is None:
        return None, None
    projected = np.asarray(homography, dtype=float) @ np.asarray([x, y, 1.0], dtype=float)
    if not np.isfinite(projected).all() or abs(projected[2]) < 1e-9:
        return None, None
    return (
        float(projected[0] / projected[2] / COURT_WIDTH_M),
        float(projected[1] / projected[2] / COURT_LENGTH_M),
    )


def _point_number(clip: str) -> int:
    return int(str(clip).removeprefix("pt"))


def _load_point_homographies(path: Path) -> dict[int, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {
            int(point): np.asarray(homography, dtype=float)
            for point, homography in zip(data["pts"], data["H"], strict=True)
            if np.isfinite(homography).all()
        }


def _merged_candidates(
    match_root: Path, names: tuple[str, ...]
) -> dict[str, dict[int, list[tuple[float, float, float]]]]:
    merged: dict[str, dict[int, list[tuple[float, float, float]]]] = {}
    for name in names:
        for clip, by_frame in _read_native_rows(match_root / name).items():
            for frame, values in by_frame.items():
                merged.setdefault(clip, {}).setdefault(frame, []).extend(values)
    return merged


def broadcast_alternate_rows(
    root: Path,
    broadcast: str,
    *,
    event_track: str = LEGACY_EVENT_TRACK,
    live_shot_camera: bool = False,
) -> list[ShardRow]:
    """Every alternate proposal of one broadcast, over its active-play frames."""

    match_root = root / broadcast
    frames_by_clip = scoped_frames(root, broadcast, live_shot_camera=live_shot_camera)
    if not frames_by_clip:
        return []
    track, scale_x, scale_y, _space = bound_crop_track(match_root, event_track=event_track)
    arc_rows = _read_native_rows(match_root / ARC_TRACK_NAME)
    far_rows = _merged_candidates(match_root, FAR_NATIVE_CANDIDATES)
    sliding_rows = _merged_candidates(match_root, SLIDING_CANDIDATES)
    output: list[ShardRow] = []
    for clip, frames in sorted(frames_by_clip.items()):
        observations = track.get(clip, {})
        centre_x, centre_y, observed = track_centres(observations, frames, scale_x, scale_y)
        arc_track = {}
        for frame, values in (arc_rows.get(clip) or {}).items():
            point = _best_candidate(values)
            if point is not None:
                arc_track[frame] = point
        output.extend(
            alternate_centres(
                match_root,
                clip,
                frames,
                centre_x,
                centre_y,
                observed,
                arc_track=arc_track,
                far_native=far_rows.get(clip),
                sliding=sliding_rows.get(clip),
            )
        )
    return output


def proposal_crop_rows(
    root: Path,
    broadcast: str,
    proposal_document: Path,
    *,
    event_track: str = LEGACY_EVENT_TRACK,
) -> list[ShardRow]:
    """Make extra classifier rows for an explicit proposal artifact.

    This intentionally accepts an explicit path rather than looking for a
    magic file under ``root``: proposal selection is a pipeline input, and an
    ambient stale proposal receipt must not change inference.  The composed
    track supplies the crop centre; an unplaceable proposal abstains here.
    """

    from cv.pipeline.event_proposals import load_proposal_document

    match_root = root / broadcast
    if event_track == LEGACY_EVENT_TRACK and not (match_root / TRACK_NAME).exists():
        return []
    tracks, scale_x, scale_y, _space = bound_crop_track(match_root, event_track=event_track)
    proposals = load_proposal_document(proposal_document)
    output: list[ShardRow] = []
    kept_by_clip: dict[str, list[tuple[int, float, float]]] = {}
    prefix = f"{broadcast}__"
    for proposal in sorted(proposals, key=lambda row: (row.clip, row.frame, -row.confidence)):
        if not proposal.clip.startswith(prefix):
            continue
        clip = proposal.clip.removeprefix(prefix)
        frame = int(round(proposal.frame))
        observations = tracks.get(clip, {})
        observed = frame in observations
        # The proposer's own predicted pixel is the point of the arm: at an
        # occlusion the composed track has nothing to centre on, and the two
        # local flight models say where the ball was.  It is a prediction and
        # is written as an unobserved row, never as a detection.
        predicted = _proposal_centre(proposal)
        if predicted is None:
            point = observations.get(frame)
            if point is None:
                continue
            predicted = (float(point[0]) * scale_x, float(point[1]) * scale_y)
        if not all(math.isfinite(value) for value in predicted):
            continue
        centre_x, centre_y, _ = track_centres(observations, [frame], scale_x, scale_y)
        base = (float(centre_x[0]), float(centre_y[0]))
        # The base store already cuts one crop on every active-play frame from
        # the interpolated composed track.  A proposal that lands on the same
        # pixel would be the same picture scored twice, so it is dropped here
        # rather than allowed to look like extra evidence.
        if observations and math.hypot(*(a - b for a, b in zip(predicted, base))) <= (
            ALTERNATE_MIN_OFFSET_PX
        ):
            continue
        if any(
            other_frame == frame
            and math.hypot(predicted[0] - x, predicted[1] - y) <= ALTERNATE_SEPARATION_PX
            for other_frame, x, y in kept_by_clip.get(clip, ())
        ):
            continue
        kept_by_clip.setdefault(clip, []).append((frame, predicted[0], predicted[1]))
        output.append(
            ShardRow(
                clip=clip,
                frame=frame,
                centre_x=predicted[0],
                centre_y=predicted[1],
                track_observed=observed,
                source=proposal.source,
                proposal_kinds=proposal.kinds,
            )
        )
    return output


def _proposal_centre(proposal) -> tuple[float, float] | None:
    """The native pixel a proposal predicts for its own epoch, when it has one."""

    evidence = proposal.evidence or {}
    x, y = evidence.get("predicted_x"), evidence.get("predicted_y")
    if x is None or y is None:
        return None
    try:
        return float(x), float(y)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# shard writing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ShardRow:
    """One row to cut: a frame and the native pixel its crop is centred on."""

    clip: str
    frame: int
    centre_x: float
    centre_y: float
    track_observed: bool
    source: str = "track"
    # This is the proposal-to-grammar entry: compatible kinds survive crop
    # cutting as metadata.  They constrain no classifier score by themselves;
    # the decoder still owns every accepted physical type.
    proposal_kinds: tuple[str, ...] = ()


def _frame_path(match_root: Path, clip: str, frame: int) -> Path:
    return match_root / FRAMES_DIR / clip / f"f_{frame:04d}.jpg"


def _cadence(match_root: Path) -> dict[str, list[int]]:
    path = match_root / CADENCE_NAME
    if not path.exists():
        return {}
    document = json.loads(path.read_text())
    return {
        str(row["clip"]): [int(value) for value in (row.get("duplicate_frames") or [])]
        for row in document.get("rows", [])
    }


def _track_space(match_root: Path) -> dict:
    track = match_root / TRACK_NAME
    with track.open(newline="") as handle:
        columns = csv.DictReader(handle).fieldnames or []
    return track_coordinate_space(match_root / f"{TRACK_NAME}.coordinates.json", list(columns))


def bound_crop_track(
    match_root: Path, *, event_track: str = LEGACY_EVENT_TRACK
) -> tuple[dict[str, dict[int, tuple[float, float]]], float, float, dict]:
    """The selected track's observations and the scale that makes them image pixels.

    Crops and numeric features resolve the *same* binding, so a frame is an
    observation in both readers or in neither and both centre on the same pixel.
    Under ``event_track="s4_current"`` the observations are the measured rows
    only: a guide-interpolated or ancestry-uncertain row still gets a crop, cut
    at the interpolated position an absent row would get, and reports
    ``track_observed=False``.
    """

    validate_event_track(event_track)
    match_root = Path(match_root)
    if event_track != LEGACY_EVENT_TRACK:
        bound = bind_broadcast_track(match_root, event_track=event_track)
        binding = bound.binding
        return (
            bound.observations,
            binding.scale_x,
            binding.scale_y,
            {**bound.space, "scale_x": binding.scale_x, "scale_y": binding.scale_y},
        )
    track_path = match_root / TRACK_NAME
    if not track_path.exists():
        return {}, 1.0, 1.0, {"space": "absent"}
    space = _track_space(match_root)
    return (
        read_track(track_path, space["x_column"], space["y_column"]),
        float(space["scale_x"]),
        float(space["scale_y"]),
        space,
    )


def _fps(match_root: Path) -> float:
    sidecar = match_root / f"{FRAMES_DIR}.coordinates.json"
    try:
        rate = float(json.loads(sidecar.read_text())["fps"])
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing or invalid source frame rate: {sidecar}") from error
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(f"source frame rate must be positive and finite: {sidecar}")
    return rate


def _source_rank(source: str) -> int:
    return ALTERNATE_SOURCES.index(source) + 1 if source in ALTERNATE_SOURCES else 0


def normalise_shard_rows(rows) -> list[ShardRow]:
    """Accept either ``(clip, frame)`` pairs or explicit-centre :class:`ShardRow`s."""

    output: list[ShardRow] = []
    for row in rows:
        if isinstance(row, ShardRow):
            output.append(row)
            continue
        clip, frame = row
        output.append(
            ShardRow(
                clip=str(clip),
                frame=int(frame),
                centre_x=math.nan,
                centre_y=math.nan,
                track_observed=False,
            )
        )
    return output


def build_broadcast_shard(
    root: Path,
    broadcast: str,
    rows,
    output: Path,
    *,
    event_track: str = LEGACY_EVENT_TRACK,
) -> dict:
    """Cut every requested row of one broadcast and write its shard.

    A row is either a ``(clip, frame)`` pair, whose crop is centred on the
    composed track, or a :class:`ShardRow` that names its own native centre,
    which is how an alternate proposal asks for a different picture of the same
    frame.
    """

    match_root = root / broadcast
    track, scale_x, scale_y, space = bound_crop_track(match_root, event_track=event_track)
    fps = _fps(match_root)
    cadence = _cadence(match_root)
    windows = point_windows(match_root)
    reel = match_root / REEL_NAME

    by_clip: dict[str, list[ShardRow]] = {}
    for row in normalise_shard_rows(rows):
        by_clip.setdefault(row.clip, []).append(row)

    total = len(rows)
    tight = np.zeros((total, SEQUENCE_LENGTH, TIGHT_HEIGHT, TIGHT_WIDTH, 3), dtype=np.uint8)
    context = np.zeros((total, CONTEXT_HEIGHT, CONTEXT_WIDTH, 3), dtype=np.uint8)
    mel = np.zeros((total, MEL_BANDS, MEL_FRAMES), dtype=np.float16)
    clips_out: list[str] = []
    frames_out: list[int] = []
    centres = np.zeros((total, 2), dtype=np.float32)
    observed_out = np.zeros(total, dtype=bool)
    audio_present = np.zeros(total, dtype=bool)
    missing_frames = 0

    cursor = 0
    sources_out: list[str] = []
    for clip in sorted(by_clip):
        shard_members = sorted(by_clip[clip], key=lambda row: (row.frame, _source_rank(row.source)))
        frames = [row.frame for row in shard_members]
        directory = match_root / FRAMES_DIR / clip
        available = sorted(int(path.stem.removeprefix("f_")) for path in directory.glob("f_*.jpg"))
        n_frames = available[-1] if available else 0
        exposures = unique_exposures(cadence.get(clip, []), n_frames)
        observations = track.get(clip, {})
        centre_x, centre_y, observed = track_centres(observations, frames, scale_x, scale_y)
        for position, row in enumerate(shard_members):
            if math.isfinite(row.centre_x) and math.isfinite(row.centre_y):
                centre_x[position] = np.float32(row.centre_x)
                centre_y[position] = np.float32(row.centre_y)
                observed[position] = row.track_observed

        needed: set[int] = set()
        plan = []
        for position, frame in enumerate(frames):
            members = sequence_frames(exposures, frame)
            plan.append(members)
            needed.update(members)
            needed.add(frame)
        cache: dict[int, np.ndarray | None] = {}
        for frame in sorted(needed):
            path = _frame_path(match_root, clip, frame)
            image = cv2.imread(str(path), cv2.IMREAD_COLOR) if path.exists() else None
            if image is None:
                missing_frames += 1
            cache[frame] = image

        span = windows.get(clip)
        samples = None
        audio_origin = 0.0
        if span is not None and reel.exists():
            pad = AUDIO_SECONDS
            audio_origin = max(span[0] - pad, 0.0)
            samples = read_audio(
                reel, audio_origin, (span[1] - span[0]) + (span[0] - audio_origin) + pad
            )

        for position, frame in enumerate(frames):
            left, top = crop_bounds(
                float(centre_x[position]), float(centre_y[position]), TIGHT_WIDTH, TIGHT_HEIGHT
            )
            context_left, context_top = crop_bounds(
                float(centre_x[position]),
                float(centre_y[position]),
                CONTEXT_WIDTH,
                CONTEXT_HEIGHT,
            )
            for step, member in enumerate(plan[position]):
                tight[cursor, step] = cut(cache.get(member), left, top, TIGHT_WIDTH, TIGHT_HEIGHT)
            context[cursor] = cut(
                cache.get(frame), context_left, context_top, CONTEXT_WIDTH, CONTEXT_HEIGHT
            )
            if samples is not None and samples.size:
                centre_time = (span[0] - audio_origin) + (frame - 1) / fps
                start = int(round((centre_time - AUDIO_SECONDS / 2.0) * SAMPLE_RATE))
                stop = start + int(round(AUDIO_SECONDS * SAMPLE_RATE))
                if start >= 0 and stop <= samples.size:
                    mel[cursor] = log_mel(samples[start:stop]).astype(np.float16)
                    audio_present[cursor] = True
            centres[cursor] = (float(centre_x[position]), float(centre_y[position]))
            observed_out[cursor] = bool(observed[position])
            clips_out.append(f"{broadcast}__{clip}")
            frames_out.append(int(frame))
            sources_out.append(shard_members[position].source)
            cursor += 1
        cache.clear()

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        tight=tight,
        context=context,
        mel=mel,
        clips=np.asarray(clips_out),
        frames=np.asarray(frames_out, dtype=np.int32),
        centres=centres,
        track_observed=observed_out,
        audio_present=audio_present,
        proposal_source=np.asarray(sources_out),
    )
    return {
        "broadcast": broadcast,
        "rows": total,
        "clips": len(by_clip),
        "path": str(output),
        "bytes": output.stat().st_size,
        "fps": fps,
        "missing_source_frames": missing_frames,
        "audio_rows": int(audio_present.sum()),
        "track_observed_rows": int(observed_out.sum()),
        "track_coordinate_space": space,
        "proposal_sources": {name: sources_out.count(name) for name in sorted(set(sources_out))},
    }


def _shard_worker(payload: tuple[str, str, list[tuple[str, int]], str, str]) -> dict:
    root, broadcast, rows, output, event_track = payload
    return build_broadcast_shard(Path(root), broadcast, rows, Path(output), event_track=event_track)


# --------------------------------------------------------------------------- #
# uncompressed-npz memory mapping
# --------------------------------------------------------------------------- #
def memmap_member(path: Path, name: str) -> np.ndarray:
    """Memory-map one stored (uncompressed) member of a ``.npz`` archive.

    The shards are written with :func:`numpy.savez`, so every member is a stored
    zip entry holding a ``.npy`` payload; mapping it avoids materialising tens of
    gigabytes per training fold.
    """

    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(f"{name}.npy")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{path}:{name} is compressed and cannot be mapped")
        header_offset = info.header_offset
    with path.open("rb") as handle:
        handle.seek(header_offset)
        local = handle.read(30)
        if local[:4] != b"PK\x03\x04":
            raise ValueError(f"{path}:{name} has no local file header")
        name_length, extra_length = struct.unpack("<HH", local[26:30])
        data_offset = header_offset + 30 + name_length + extra_length
        handle.seek(data_offset)
        magic = handle.read(8)
        if magic[:6] != b"\x93NUMPY":
            raise ValueError(f"{path}:{name} is not a .npy payload")
        if magic[6] == 1:
            (header_length,) = struct.unpack("<H", handle.read(2))
            payload_offset = data_offset + 10 + header_length
        else:
            (header_length,) = struct.unpack("<I", handle.read(4))
            payload_offset = data_offset + 12 + header_length
        header = ast.literal_eval(handle.read(header_length).decode("latin1"))
    return np.memmap(
        path,
        dtype=np.dtype(header["descr"]),
        mode="r",
        offset=payload_offset,
        shape=tuple(header["shape"]),
    )


# --------------------------------------------------------------------------- #
# manifest and truth join
# --------------------------------------------------------------------------- #
def crop_input_paths(
    root: Path,
    *,
    alternate: bool = False,
    court_geometry: str = "point_static",
    event_track: str = LEGACY_EVENT_TRACK,
) -> list[Path]:
    """Inventory crop/feature sources, including the presence of optional witnesses.

    Do not hash the entire root: it can contain this stage's generated cache.

    Under a non-legacy ``event_track`` the selected track, its sidecar, the guide
    CSV its sidecar declares and that guide's own sidecar all join the inventory,
    so any byte change in the bound ancestry -- or a change of selection --
    invalidates the crop cache instead of silently reusing legacy pictures.
    """
    validate_event_track(event_track)
    active_path = root / "active_play_v1.json"
    active = json.loads(active_path.read_text())
    broadcasts = sorted({point_key.split("/", 1)[0] for point_key in active})
    names = [
        TRACK_NAME,
        f"{TRACK_NAME}.coordinates.json",
        "court_H_per_point.npz",
        FRAMES_DIR,
        f"{FRAMES_DIR}.coordinates.json",
        CADENCE_NAME,
        AUDIO_CACHE_NAME,
        "audit_reel_point_map.csv",
        REEL_NAME,
    ]
    if court_geometry not in COURT_GEOMETRY_MODES:
        raise ValueError(f"unsupported court_geometry mode: {court_geometry!r}")
    if court_geometry == "reliable_per_frame":
        names.extend((FRAME_HOMOGRAPHY_NAME, f"{FRAME_HOMOGRAPHY_NAME}.coordinates.json"))
    if alternate:
        for name in (ARC_TRACK_NAME, *FAR_NATIVE_CANDIDATES, *SLIDING_CANDIDATES):
            names.extend((name, f"{name}.coordinates.json"))
    paths = [active_path]
    for broadcast in broadcasts:
        if not broadcast or broadcast in (".", "..") or Path(broadcast).name != broadcast:
            raise ValueError(f"invalid broadcast key: {broadcast!r}")
        paths.extend(path for name in names if (path := root / broadcast / name).exists())
        if event_track != LEGACY_EVENT_TRACK:
            known = set(paths)
            paths.extend(
                path for path in current_track_inputs(root / broadcast) if path not in known
            )
    return paths


def ensure_runtime_dataset(
    root: Path,
    output: Path,
    jobs: int = 8,
    *,
    reuse: bool = True,
    alternate: bool = False,
    court_geometry: str = "point_static",
    court_frame_missing: str = "hold",
    event_track: str = LEGACY_EVENT_TRACK,
    live_shot_camera: bool = False,
) -> dict:
    """Reuse only crops whose actual inputs, implementation and shards still match."""
    validate_court_transport(court_geometry, court_frame_missing)
    validate_event_track(event_track)
    transport_args = (
        {"court_geometry": court_geometry, "court_frame_missing": court_frame_missing}
        if court_geometry != "point_static"
        else {}
    )
    # A selected track joins the identity configuration only when it is not the
    # default, exactly as the transport arguments do: a default run's receipt
    # keeps its shape, and any non-legacy selection is a different cache entry.
    track_args = {"event_track": event_track} if event_track != LEGACY_EVENT_TRACK else {}
    # A default crop cache must keep its identity. The switch is a different cache.
    shot_args = {"live_shot_camera": True} if live_shot_camera else {}
    stage = "event_crops_alternate" if alternate else "event_crops"
    expected_schema = ALTERNATE_SCHEMA if alternate else SCHEMA
    builder = build_alternate_dataset if alternate else build_dataset
    identity_args = {
        "stage": stage,
        "command": ["python", "-m", "cv.pipeline.event_crops"],
        "configuration": {
            "entrypoint": "build_alternate_dataset" if alternate else "build_dataset",
            "root": str(root.resolve()),
            **transport_args,
            **track_args,
            **shot_args,
        },
        "inputs": crop_input_paths(
            root, alternate=alternate, court_geometry=court_geometry, event_track=event_track
        ),
    }
    manifest_path = output / "manifest.json"

    def outputs(manifest: dict) -> list[Path]:
        if manifest["schema"] != expected_schema:
            raise ValueError("unexpected crop schema")
        shards = [Path(row["path"]) for row in manifest["shards"]]
        if any(not path.resolve().is_relative_to(output.resolve()) for path in shards):
            raise ValueError("crop shard is outside its output directory")
        return [manifest_path, *shards]

    if reuse:
        try:
            manifest = json.loads(manifest_path.read_text())
            if stage_receipt_matches(out_dir=output, outputs=outputs(manifest), **identity_args):
                return {"reused": True, "receipt": str(receipt_path(output, stage))}
        except (OSError, ValueError, KeyError, TypeError):
            pass
    before = stage_identity(**identity_args)
    manifest = builder(root, output, jobs, **transport_args, **track_args, **shot_args)
    # Re-enumerate optional inputs as well as hashing again: a new witness can
    # otherwise appear during construction without changing any original file.
    identity_args["inputs"] = crop_input_paths(
        root, alternate=alternate, court_geometry=court_geometry, event_track=event_track
    )
    receipt = write_stage_receipt(
        out_dir=output, outputs=outputs(manifest), expected_identity=before, **identity_args
    )
    return {"reused": False, "receipt": str(receipt)}


def build_dataset(
    root: Path,
    output: Path,
    jobs: int = 8,
    broadcasts: list[str] | None = None,
    *,
    court_geometry: str = "point_static",
    court_frame_missing: str = "hold",
    event_track: str = LEGACY_EVENT_TRACK,
    live_shot_camera: bool = False,
) -> dict:
    """Build every shard for ``root`` and write ``manifest.json`` beside them."""

    validate_court_transport(court_geometry, court_frame_missing)
    validate_event_track(event_track)
    transport_args = (
        {"court_geometry": court_geometry, "court_frame_missing": court_frame_missing}
        if court_geometry != "point_static"
        else {}
    )
    dataset, feature_manifest = build_automatic_dataset(
        root, event_track=event_track, live_shot_camera=live_shot_camera, **transport_args
    )
    output.mkdir(parents=True, exist_ok=True)
    court_y = dataset.windows[:, RADIUS, 9].astype(np.float32)
    court_x = dataset.windows[:, RADIUS, 8].astype(np.float32)
    selected = sorted(set(dataset.broadcasts.tolist()))
    if broadcasts:
        selected = [name for name in selected if name in set(broadcasts)]

    payloads = []
    for broadcast in selected:
        mask = dataset.broadcasts == broadcast
        rows = [
            (str(clip).split("__", 1)[1], int(frame))
            for clip, frame in zip(dataset.clips[mask], dataset.frames[mask], strict=True)
        ]
        payloads.append((str(root), broadcast, rows, str(output / f"{broadcast}.npz"), event_track))

    reports = []
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for report in pool.map(_shard_worker, payloads):
                reports.append(report)
    else:
        for payload in payloads:
            reports.append(_shard_worker(payload))

    order: list[dict] = []
    for broadcast in selected:
        mask = dataset.broadcasts == broadcast
        for clip, frame, cx, gy in zip(
            dataset.clips[mask],
            dataset.frames[mask],
            court_x[mask],
            court_y[mask],
            strict=True,
        ):
            order.append(
                {
                    "broadcast": broadcast,
                    "clip": str(clip),
                    "frame": int(frame),
                    "court_x": None if not math.isfinite(float(cx)) else float(cx),
                    "court_y": None if not math.isfinite(float(gy)) else float(gy),
                }
            )
    if court_geometry != "point_static":
        transports = {
            name: load_court_transport(
                root / name, mode=court_geometry, missing=court_frame_missing
            )
            for name in selected
        }
        for row in order:
            row["court_geometry"] = transports[row["broadcast"]].row_record(
                row["clip"].split("__", 1)[1], row["frame"]
            )
    manifest = {
        "schema": SCHEMA,
        "root": str(root.resolve()),
        "rows": len(order),
        "shards": reports,
        "tight_shape": [SEQUENCE_LENGTH, TIGHT_HEIGHT, TIGHT_WIDTH, 3],
        "context_shape": [CONTEXT_HEIGHT, CONTEXT_WIDTH, 3],
        "mel_shape": [MEL_BANDS, MEL_FRAMES],
        "sequence_offsets": "unique exposures at -8..+7 around the centre frame",
        "audio": {
            "sample_rate": SAMPLE_RATE,
            "seconds": AUDIO_SECONDS,
            "bands": MEL_BANDS,
            "hop_samples": MEL_HOP,
            "fft_samples": MEL_FFT,
        },
        "feature_manifest": feature_manifest,
        "labels_or_reviewed_inputs": [],
        "rows_index": order,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _selected_track_record(root: Path, broadcasts: list[str], event_track: str) -> dict:
    """Declare the bound track and guide of every broadcast an alternate store cut."""

    return {
        "event_track": event_track,
        "per_broadcast": {
            broadcast: bind_broadcast_track(
                root / broadcast, event_track=event_track
            ).binding.record()
            for broadcast in sorted(broadcasts)
        },
    }


def build_alternate_dataset(
    root: Path,
    output: Path,
    jobs: int = 8,
    broadcasts: list[str] | None = None,
    proposal_document: Path | None = None,
    include_track_alternates: bool = True,
    *,
    court_geometry: str = "point_static",
    court_frame_missing: str = "hold",
    event_track: str = LEGACY_EVENT_TRACK,
    live_shot_camera: bool = False,
) -> dict:
    """Build the shards holding only the alternate proposals of ``root``.

    The base dataset centres every crop on the composed track.  This one cuts a
    second picture of the same frame wherever the track has no row of its own
    and another witness -- the arc-augmented track, the far-court native
    detector, or the sliding-window detector -- names a different pixel.  It is
    written as its own directory so the frozen base shards are reused unchanged;
    the runtime scores both and hands the union to one decoder.
    """

    validate_court_transport(court_geometry, court_frame_missing)
    validate_event_track(event_track)
    active = json.loads((root / "active_play_v1.json").read_text())
    transport_receipts = {}
    selected = sorted({point_key.split("/", 1)[0] for point_key in active})
    if broadcasts:
        selected = [name for name in selected if name in set(broadcasts)]
    output.mkdir(parents=True, exist_ok=True)

    payloads = []
    order: list[dict] = []
    for broadcast in selected:
        # A proposal-only store is scored beside the shipped alternate store,
        # not instead of it, so the two arms stay separable in the union.
        rows = (
            broadcast_alternate_rows(
                root, broadcast, event_track=event_track, live_shot_camera=live_shot_camera
            )
            if include_track_alternates
            else []
        )
        if proposal_document is not None:
            rows.extend(
                proposal_crop_rows(root, broadcast, proposal_document, event_track=event_track)
            )
        if not rows:
            continue
        rows.sort(key=lambda row: (row.clip, row.frame, _source_rank(row.source)))
        homographies = _load_point_homographies(root / broadcast / "court_H_per_point.npz")
        transport = (
            load_court_transport(root / broadcast, mode=court_geometry, missing=court_frame_missing)
            if court_geometry != "point_static"
            else None
        )
        if transport is not None:
            transport_receipts[broadcast] = {"artifact": transport.artifact}
        for row in rows:
            point = _point_number(row.clip)
            matrix = (
                transport.matrix(row.clip, int(row.frame))
                if transport is not None
                else homographies.get(point)
            )
            court_x, court_y = _court_normalised(
                {point: matrix} if matrix is not None else {}, point, row.centre_x, row.centre_y
            )
            order.append(
                {
                    "broadcast": broadcast,
                    "clip": f"{broadcast}__{row.clip}",
                    "frame": int(row.frame),
                    "court_x": court_x,
                    "court_y": court_y,
                    "proposal_source": row.source,
                    "proposal_kinds": list(row.proposal_kinds),
                    **(
                        {"court_geometry": transport.row_record(row.clip, int(row.frame))}
                        if transport is not None
                        else {}
                    ),
                }
            )
        payloads.append((str(root), broadcast, rows, str(output / f"{broadcast}.npz"), event_track))

    reports = []
    if jobs > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for report in pool.map(_shard_worker, payloads):
                reports.append(report)
    else:
        for payload in payloads:
            reports.append(_shard_worker(payload))

    manifest = {
        "schema": ALTERNATE_SCHEMA,
        **(
            {
                "court_transport": {
                    "mode": court_geometry,
                    "missing_policy": court_frame_missing,
                    "artifact_name": FRAME_HOMOGRAPHY_NAME,
                    "per_broadcast": transport_receipts,
                }
            }
            if court_geometry != "point_static"
            else {}
        ),
        **(
            {"ball_track": _selected_track_record(root, selected, event_track)}
            if event_track != LEGACY_EVENT_TRACK
            else {}
        ),
        "root": str(root.resolve()),
        "rows": len(order),
        "shards": reports,
        "tight_shape": [SEQUENCE_LENGTH, TIGHT_HEIGHT, TIGHT_WIDTH, 3],
        "context_shape": [CONTEXT_HEIGHT, CONTEXT_WIDTH, 3],
        "mel_shape": [MEL_BANDS, MEL_FRAMES],
        "sequence_offsets": "unique exposures at -8..+7 around the centre frame",
        "audio": {
            "sample_rate": SAMPLE_RATE,
            "seconds": AUDIO_SECONDS,
            "bands": MEL_BANDS,
            "hop_samples": MEL_HOP,
            "fft_samples": MEL_FFT,
        },
        "proposal": {
            "sources": list(ALTERNATE_SOURCES),
            "minimum_offset_px": ALTERNATE_MIN_OFFSET_PX,
            "separation_px": ALTERNATE_SEPARATION_PX,
            "corner_min_deg": CORNER_MIN_DEG,
            "scope": "active-play frames whose composed track has no row of its own",
            "proposal_document": (
                str(proposal_document.resolve()) if proposal_document is not None else None
            ),
            "include_track_alternates": include_track_alternates,
            "grammar_entry": (
                "proposal_kinds are compatible physical types carried into the crop index; "
                "classifier and event grammar remain the only accepting authorities"
            ),
        },
        "per_source": {
            name: sum(1 for row in order if row["proposal_source"] == name)
            for name in ALTERNATE_SOURCES
        },
        "labels_or_reviewed_inputs": [],
        "rows_index": order,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def join_truth(manifest: dict, truth: list[dict]) -> dict:
    """Attach truth type and sub-frame offset to a manifest built without labels.

    A truth event is anchored to the manifest row of its clip whose integer frame
    is nearest, within half a frame, which is the join the current classifier's
    benchmark uses.  Rows within ``2`` frames of an anchored event but not the
    anchor itself are marked ``ignore`` so a near miss is not trained as a
    negative.
    """

    index: dict[tuple[str, int], int] = {}
    by_clip: dict[str, list[int]] = {}
    for position, row in enumerate(manifest["rows_index"]):
        index[(row["clip"], row["frame"])] = position
        by_clip.setdefault(row["clip"], []).append(row["frame"])
    types = ["none"] * len(manifest["rows_index"])
    offsets = [0.0] * len(manifest["rows_index"])
    ignore = [False] * len(manifest["rows_index"])
    assigned = 0
    outside = 0
    for event in truth:
        candidates = by_clip.get(event["clip"], [])
        if not candidates:
            outside += 1
            continue
        frame = min(candidates, key=lambda value: (abs(value - event["frame"]), value))
        if abs(frame - event["frame"]) > TRUTH_ANCHOR_TOLERANCE:
            outside += 1
            continue
        position = index[(event["clip"], frame)]
        if types[position] not in {"none", event["event_type"]}:
            raise ValueError(f"conflicting truth at {event['clip']} f{frame}")
        types[position] = event["event_type"]
        offsets[position] = float(event["frame"]) - float(frame)
        assigned += 1
        for neighbour in range(frame - 2, frame + 3):
            other = index.get((event["clip"], neighbour))
            if other is not None and other != position:
                ignore[other] = True
    return {
        "truth_type": types,
        "truth_offset": offsets,
        "ignore": ignore,
        "assigned": assigned,
        "outside_automatic_event_scope": outside,
        "truth_events": len(truth),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--court-geometry", choices=COURT_GEOMETRY_MODES, default="point_static")
    parser.add_argument(
        "--court-frame-missing", choices=COURT_FRAME_MISSING_POLICIES, default="hold"
    )
    parser.add_argument(
        "--event-track",
        choices=(LEGACY_EVENT_TRACK, CURRENT_EVENT_TRACK),
        default=LEGACY_EVENT_TRACK,
        help="which ball track supplies the crop centres and the observed mask",
    )
    parser.add_argument("--broadcast", action="append", dest="broadcasts")
    parser.add_argument(
        "--alternate-proposals",
        action="store_true",
        help="cut only the alternate proposal rows: a second crop of the frames "
        "the composed track cannot centre",
    )
    parser.add_argument(
        "--proposal-document",
        type=Path,
        help="explicit label-free event_proposals_v1 receipt to union with alternate crops",
    )
    parser.add_argument(
        "--proposals-only",
        action="store_true",
        help="cut the proposal rows alone, so the arm is separable from the shipped "
        "detector-alternate store it is scored beside",
    )
    args = parser.parse_args()
    if args.proposal_document is not None and not args.alternate_proposals:
        parser.error("--proposal-document requires --alternate-proposals")
    builder = build_alternate_dataset if args.alternate_proposals else build_dataset
    if args.alternate_proposals:
        manifest = builder(
            args.root,
            args.output,
            args.jobs,
            args.broadcasts,
            proposal_document=args.proposal_document,
            include_track_alternates=not args.proposals_only,
            court_geometry=args.court_geometry,
            court_frame_missing=args.court_frame_missing,
            event_track=args.event_track,
        )
    else:
        manifest = builder(
            args.root,
            args.output,
            args.jobs,
            args.broadcasts,
            court_geometry=args.court_geometry,
            court_frame_missing=args.court_frame_missing,
            event_track=args.event_track,
        )
    summary = {key: value for key, value in manifest.items() if key != "rows_index"}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
