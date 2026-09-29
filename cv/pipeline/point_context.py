"""Build a source-only context prior around each active-play segment.

This module is the automatic event stage's look-ahead sidecar.  It does not
decode physical events and never opens labels.  It consumes explicitly named
active-play segments, player boxes/court geometry, event emissions, native score
frames, and aligned broadcast audio.  Missing or sampled-away context abstains.

The tentative event-grammar lane may consume one JSON file per point with this
versioned contract (all likelihood maps are normalized non-negative relative
weights and are evidence, never hard labels)::

    {
      "schema": "tennis_point_context_prior_v1",
      "automatic": true,
      "match_id": "...", "point_id": "pt0001", "segment_index": 0,
      "serve_ordinal": {"first": .25, "second": .25,
                         "fault": .25, "let": .25},
      "ending_kind": {"out": .25, "second_bounce": .25,
                       "net": .25, "fov_exit": .25},
      "winner": "server|receiver|unknown",
      "winner_likelihoods": {"server": 0, "receiver": 0, "unknown": 1},
      "evidence_receipts": {
        "segment_adjacency": {...}, "scoreboard": {...},
        "audio_call": {...}, "shipped_ending": {...}
      },
      "abstentions": ["..."],
      "provenance": {...}
    }

Required consumer behavior: treat absent files, another schema version,
``automatic != true``, non-normalized/non-finite likelihoods, or receipts naming
human-derived input as unavailable context.  A consumer may add bounded log
evidence from these maps but must preserve its own abstention and must never
reinterpret the argmax as truth.  ``winner`` is ``unknown`` unless a legal score
transition and server-row witness jointly determine the role.  ``fault`` and
``let`` describe the current attempt's terminal service outcome; ``first`` and
``second`` describe its ordinal, so they are intentionally represented in one
four-way interface requested by the event grammar rather than as two independent
categorical claims.

The audio receipt reports a voice-band burst after a candidate ending.  Its
``out_call_likelihood_ratio`` is a calibrated soft witness; it is not speech
recognition and crowd noise can produce a large value.  Native frame IDs and the
audio/video mapping are retained without inventing exposures.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import cv2

from cv.pipeline import resolution as res
import numpy as np

from cv.pipeline import paths, provenance, score_vlm, serve_speed_graphic
from cv.pipeline.score_grammar import award_point, state_from_read

SCHEMA = "tennis_point_context_prior_v1"
COURT_WIDTH_M = 10.97
COURT_LENGTH_M = 23.77
AUDIO_SAMPLE_RATE = 16_000
AUDIO_FFT_SAMPLES = 640
AUDIO_HOP_SAMPLES = 160
VOICE_BAND_HZ = (250.0, 3_400.0)
CALL_POST_SECONDS = (0.2, 1.5)
CALL_BASELINE_SECONDS = (-1.2, -0.1)


@lru_cache(maxsize=32)
def _cached_file_record(path: str) -> dict[str, Any]:
    """Hash large source videos once per process, not once per point."""
    return provenance.file_record(Path(path))


@dataclass(frozen=True)
class Segment:
    """One ordered active-play point/attempt window."""

    match_id: str
    point_id: str
    fps: float
    start_frame: float
    end_frame: float
    source_start_seconds: float | None = None
    source_end_seconds: float | None = None
    source_timeline_available: bool = False


@dataclass(frozen=True)
class ServerPosition:
    end: str | None
    side: str | None
    court_x: float | None
    court_y: float | None
    contact_frame: float | None
    confidence: float
    abstention_reason: str | None
    evidence: dict[str, Any]


@dataclass(frozen=True)
class CallCalibration:
    """Fixed logistic calibration for the source-only voice-burst statistic."""

    midpoint: float = 5.0
    scale: float = 1.5
    maximum_likelihood_ratio: float = 20.0


def normalized_likelihoods(values: Mapping[str, float], keys: Sequence[str]) -> dict[str, float]:
    numbers = np.asarray([float(values.get(key, 0.0)) for key in keys], dtype=float)
    if not np.all(np.isfinite(numbers)) or np.any(numbers < 0):
        raise ValueError("likelihoods must be finite and non-negative")
    total = float(numbers.sum())
    if total <= 0:
        numbers[:] = 1.0 / len(numbers)
    else:
        numbers /= total
    return {key: float(value) for key, value in zip(keys, numbers, strict=True)}


def load_active_segments(path: Path, *, source_timeline: bool = False) -> list[Segment]:
    """Load the active-play mapping without assuming audit-reel gaps are real gaps."""
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise ValueError("active-play artifact must be a mapping")
    output = []
    for key, row in document.items():
        if not isinstance(row, dict) or "/" not in key:
            continue
        match_id, point_id = key.rsplit("/", 1)
        spans = row.get("active_spans") or []
        if not spans:
            continue
        start = min(float(span[0]) for span in spans)
        end = max(float(span[1]) for span in spans)
        fps = float(row.get("fps", 0.0))
        if not math.isfinite(fps) or fps <= 0 or end < start:
            continue
        output.append(
            Segment(
                match_id=match_id,
                point_id=point_id,
                fps=fps,
                start_frame=start,
                end_frame=end,
                source_timeline_available=source_timeline,
            )
        )
    return sorted(output, key=lambda row: (row.match_id, _point_number(row.point_id)))


def _source_timeline_segments(
    segments: Sequence[Segment], match_dir: Path, requested: bool | None
) -> list[Segment]:
    """Hydrate audit-local spans with original times when a raw source is present.

    Raw pipeline artifacts expose the source video as the audit-reel symlink and
    retain each point's absolute ``rally_t_*`` values in the point map.  Older
    evaluation cohorts instead contain a real concatenated reel.  Auto mode
    therefore selects source time only for the former and safely falls back for
    the latter.
    """
    reel = match_dir / "audit_reel_native_1080.mp4"
    point_map = match_dir / "audit_reel_point_map.csv"
    source_backed = reel.is_symlink() and reel.resolve().is_file() and point_map.is_file()
    enabled = source_backed if requested is None else requested and source_backed
    if not enabled:
        return [replace(segment, source_timeline_available=False) for segment in segments]
    with point_map.open(newline="") as handle:
        rows = {str(row.get("pt")): row for row in csv.DictReader(handle)}
    output = []
    for segment in segments:
        row = rows.get(str(_point_number(segment.point_id)[0]))
        try:
            offset = float(row["rally_t_start"]) if row else None
        except (KeyError, TypeError, ValueError):
            offset = None
        if offset is None:
            output.append(replace(segment, source_timeline_available=False))
            continue
        output.append(
            replace(
                segment,
                source_start_seconds=offset + (segment.start_frame - 1.0) / segment.fps,
                source_end_seconds=offset + (segment.end_frame - 1.0) / segment.fps,
                source_timeline_available=True,
            )
        )
    return output


def _point_number(point_id: str) -> tuple[int, str]:
    digits = "".join(character for character in point_id if character.isdigit())
    return (int(digits) if digits else 10**9, point_id)


def adjacency_windows(segments: Sequence[Segment], radius: int = 3) -> dict[tuple[str, str], dict]:
    """Return previous/next one-to-three segments, never crossing a broadcast."""
    if radius not in (1, 2, 3):
        raise ValueError("adjacency radius must be one to three")
    output = {}
    by_match: dict[str, list[Segment]] = {}
    for segment in segments:
        by_match.setdefault(segment.match_id, []).append(segment)
    for match_id, rows in by_match.items():
        rows.sort(key=lambda row: _point_number(row.point_id))
        for index, row in enumerate(rows):
            output[(match_id, row.point_id)] = {
                "previous": [asdict(value) for value in rows[max(0, index - radius) : index]],
                "next": [asdict(value) for value in rows[index + 1 : index + radius + 1]],
            }
    return output


def load_emissions(path: Path) -> dict[tuple[str, str], list[dict]]:
    document = json.loads(path.read_text())
    rows = document.get("emissions") if isinstance(document, dict) else document
    if not isinstance(rows, list):
        raise ValueError("emission artifact must contain a list")
    output: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        clip = str(row.get("clip", ""))
        match_id = str(row.get("match_id", ""))
        if "__" in clip:
            match_id, point_id = clip.rsplit("__", 1)
        else:
            point_id = clip
        if not match_id or not point_id:
            continue
        output.setdefault((match_id, point_id), []).append(row)
    for values in output.values():
        values.sort(key=lambda row: (float(row.get("frame", 0)), str(row.get("event_type"))))
    return output


def _box_distance(pixel: tuple[float, float], row: Mapping[str, str]) -> float:
    """Distance from a native pixel to a box, in box heights, in the sidecar-declared scale."""
    scale_x = float(row.get("_scale_x", 1.0))
    scale_y = float(row.get("_scale_y", 1.0))
    x0, x1 = (scale_x * float(row[key]) for key in ("x0", "x1"))
    y0, y1 = (scale_y * float(row[key]) for key in ("y0", "y1"))
    dx = max(x0 - pixel[0], 0.0, pixel[0] - x1)
    dy = max(y0 - pixel[1], 0.0, pixel[1] - y1)
    return float(math.hypot(dx, dy) / max(y1 - y0, 1.0))


def infer_server_position(
    segment: Segment,
    events: Sequence[dict],
    player_rows: Sequence[Mapping[str, str]],
) -> ServerPosition:
    """Resolve the serving end/side from the first contact and player court root."""
    contacts = [row for row in events if row.get("event_type") == "contact"]
    if not contacts:
        return ServerPosition(None, None, None, None, None, 0.0, "no_contact_emission", {})
    contact = contacts[0]
    frame = float(contact["frame"])
    location = contact.get("location") or {}
    if location.get("image_x") is None or location.get("image_y") is None:
        return ServerPosition(None, None, None, None, frame, 0.0, "contact_has_no_pixel", {})
    candidates = [
        row
        for row in player_rows
        if str(row.get("clip")) == segment.point_id
        and abs(_frame_number(str(row.get("frame", ""))) - frame) <= 2.0
        and row.get("court_x") not in (None, "")
        and row.get("court_y") not in (None, "")
    ]
    if not candidates:
        return ServerPosition(None, None, None, None, frame, 0.0, "no_player_court_rows", {})
    ranked = sorted(
        [
            (
                _box_distance((float(location["image_x"]), float(location["image_y"])), row),
                row,
            )
            for row in candidates
        ],
        key=lambda item: item[0],
    )
    distance, selected = ranked[0]
    if distance > 1.25:
        return ServerPosition(
            None,
            None,
            None,
            None,
            frame,
            0.0,
            "contact_not_near_player",
            {"nearest_box_heights": distance, "candidate_count": len(candidates)},
        )
    court_x, court_y = float(selected["court_x"]), float(selected["court_y"])
    if court_y < COURT_LENGTH_M / 2 - 1.0:
        end = "near"
        side = "ad" if court_x < COURT_WIDTH_M / 2 else "deuce"
    elif court_y > COURT_LENGTH_M / 2 + 1.0:
        end = "far"
        side = "ad" if court_x > COURT_WIDTH_M / 2 else "deuce"
    else:
        return ServerPosition(
            None,
            None,
            court_x,
            court_y,
            frame,
            0.0,
            "serving_player_near_net",
            {"normalized_box_distance": distance},
        )
    confidence = float(np.clip(1.0 - distance / 1.25, 0.0, 1.0))
    return ServerPosition(
        end,
        side,
        court_x,
        court_y,
        frame,
        confidence,
        None,
        {
            "method": "first emitted contact nearest player box with automatic court root",
            "normalized_box_distance": distance,
            "box_scale_to_contact_pixel": [
                float(selected.get("_scale_x", 1.0)),
                float(selected.get("_scale_y", 1.0)),
            ],
            "player_row": dict(selected),
        },
    )


def _frame_number(value: str) -> int:
    stem = Path(value).stem
    digits = "".join(character for character in stem if character.isdigit())
    return int(digits) if digits else -(10**9)


def segment_adjacency_evidence(
    current: Segment,
    following: Segment | None,
    current_server: ServerPosition,
    following_server: ServerPosition | None,
    current_events: Sequence[dict],
) -> dict[str, Any]:
    """Interpret same-side/alternate-side/end-change only on an original timeline."""
    base = {
        "schema": "tennis_segment_adjacency_evidence_v1",
        "current_server": asdict(current_server),
        "following_segment": asdict(following) if following else None,
        "following_server": asdict(following_server) if following_server else None,
        "source_timeline_available": current.source_timeline_available,
        "relation": "unknown",
        "abstained": True,
        "abstention_reason": None,
    }
    if following is None:
        return {**base, "abstention_reason": "no_following_segment"}
    if not current.source_timeline_available:
        return {**base, "abstention_reason": "sampled_audit_reel_has_no_original_gap"}
    if current_server.end is None or following_server is None or following_server.end is None:
        return {**base, "abstention_reason": "server_geometry_unavailable"}
    if current_server.end != following_server.end:
        return {**base, "relation": "server_end_changed_game_boundary", "abstained": False}
    if current_server.side != following_server.side:
        return {**base, "relation": "service_side_alternated_completed_point", "abstained": False}
    net_before_return = False
    for row in current_events:
        if row.get("event_type") == "contact" and float(row["frame"]) > float(
            current_server.contact_frame or 0
        ):
            break
        if row.get("event_type") == "net_hit":
            net_before_return = True
    gap = None
    if current.source_end_seconds is not None and following.source_start_seconds is not None:
        gap = following.source_start_seconds - current.source_end_seconds
    let_share = 0.75 if net_before_return else (0.45 if gap is not None and gap <= 8 else 0.20)
    return {
        **base,
        "relation": "same_server_same_side_fault_or_let",
        "abstained": False,
        "let_likelihood_within_fault_or_let": let_share,
        "fault_likelihood_within_fault_or_let": 1.0 - let_share,
        "timing_gap_seconds": gap,
        "serve_flight_net_hit": net_before_return,
    }


def scoreboard_diff(
    before: serve_speed_graphic.ScoreOverlayReading,
    after: serve_speed_graphic.ScoreOverlayReading,
) -> dict[str, Any]:
    """Find the one legal point transition between consecutive scoreboard reads."""
    receipt = {
        "schema": "tennis_scoreboard_diff_v1",
        "before": _score_receipt(before),
        "after": _score_receipt(after),
        "winner_row": None,
        "winner": "unknown",
        "game_ended": None,
        "abstained": True,
        "abstention_reason": None,
    }
    if before.abstained or after.abstained:
        return {**receipt, "abstention_reason": "score_overlay_unreadable"}
    raw_before = {"present": True, "rows": list(before.rows), "serving_row": before.serving_row}
    raw_after = {"present": True, "rows": list(after.rows), "serving_row": after.serving_row}
    first = score_vlm.norm_read(raw_before)
    second = score_vlm.norm_read(raw_after)
    if first is None or second is None:
        return {**receipt, "abstention_reason": "score_tokens_do_not_form_legal_rows"}
    server_rows = [before.serving_row] if before.serving_row in (1, 2) else [1, 2]
    candidates = []
    for server_row in server_rows:
        state = state_from_read(first, server_row)
        following = state_from_read(second, server_row)
        if state is None or following is None:
            continue
        for winner_index in (0, 1):
            awarded = award_point(state, winner_index)
            # Server may change when the point ends a game; compare score columns,
            # then use the awarded state for the game-boundary receipt.
            if awarded.as_read() == following.as_read():
                candidates.append((state, awarded, winner_index + 1, server_row))
    unique = {(candidate[2], candidate[3]) for candidate in candidates}
    if len(unique) != 1:
        reason = "no_legal_single_point_transition" if not unique else "server_row_ambiguous"
        return {**receipt, "abstention_reason": reason}
    state, awarded, winner_row, server_row = candidates[0]
    winner = "server" if winner_row == server_row else "receiver"
    return {
        **receipt,
        "winner_row": winner_row,
        "server_row": server_row,
        "winner": winner,
        "game_ended": awarded.games != state.games and awarded.points == (0, 0),
        "abstained": False,
        "abstention_reason": None,
    }


def scoreboard_lookahead_evidence(
    current: serve_speed_graphic.ScoreOverlayReading,
    following: Sequence[tuple[Segment, serve_speed_graphic.ScoreOverlayReading]],
) -> dict[str, Any]:
    """Use the first legal score transition among the next one-to-three segments."""
    if len(following) > 3:
        raise ValueError("scoreboard look-ahead is limited to three segments")
    attempts = []
    for offset, (segment, reading) in enumerate(following, start=1):
        evidence = scoreboard_diff(current, reading)
        attempts.append(
            {
                "offset": offset,
                "point_id": segment.point_id,
                "abstained": evidence["abstained"],
                "abstention_reason": evidence.get("abstention_reason"),
                "after": evidence.get("after"),
            }
        )
        if not evidence["abstained"]:
            return {
                **evidence,
                "lookahead_offset": offset,
                "lookahead_point_id": segment.point_id,
                "lookahead_attempts": attempts,
            }
    reason = "no_following_segment" if not attempts else "no_legal_readable_score_in_lookahead"
    return {
        "schema": "tennis_scoreboard_diff_v1",
        "before": _score_receipt(current),
        "after": attempts[-1]["after"] if attempts else None,
        "winner_row": None,
        "winner": "unknown",
        "game_ended": None,
        "abstained": True,
        "abstention_reason": reason,
        "lookahead_offset": None,
        "lookahead_point_id": None,
        "lookahead_attempts": attempts,
    }


def _score_receipt(reading: serve_speed_graphic.ScoreOverlayReading) -> dict[str, Any]:
    return {
        **asdict(reading),
        "words": [asdict(word) for word in reading.words],
    }


def decode_audio(video: Path, start: float, duration: float) -> np.ndarray:
    """Decode mono PCM without changing the source timeline."""
    if start < 0 or duration <= 0:
        raise ValueError("non-negative audio start and positive duration required")
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-ss",
            str(start),
            "-i",
            str(video),
            "-t",
            str(duration),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(AUDIO_SAMPLE_RATE),
            "-f",
            "s16le",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
        timeout=60,
    )
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def voice_spectrogram(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return log power, voice-band log energy, and bin-centre seconds."""
    if len(samples) < AUDIO_FFT_SAMPLES:
        return np.empty((0, 0)), np.empty(0), np.empty(0)
    window = np.hanning(AUDIO_FFT_SAMPLES)
    offsets = range(0, len(samples) - AUDIO_FFT_SAMPLES + 1, AUDIO_HOP_SAMPLES)
    spectra = np.stack(
        [
            np.abs(np.fft.rfft(samples[offset : offset + AUDIO_FFT_SAMPLES] * window)) ** 2
            for offset in offsets
        ]
    )
    frequencies = np.fft.rfftfreq(AUDIO_FFT_SAMPLES, 1.0 / AUDIO_SAMPLE_RATE)
    band = (frequencies >= VOICE_BAND_HZ[0]) & (frequencies <= VOICE_BAND_HZ[1])
    log_power = np.log1p(spectra).astype(np.float32)
    voice = np.log1p(spectra[:, band].sum(axis=1)).astype(np.float32)
    seconds = (
        np.arange(len(spectra)) * AUDIO_HOP_SAMPLES + 0.5 * AUDIO_FFT_SAMPLES
    ) / AUDIO_SAMPLE_RATE
    return log_power, voice, seconds


def audio_call_witness(
    samples: np.ndarray,
    *,
    fps: float,
    candidate_frame: float,
    calibration: CallCalibration = CallCalibration(),
    render_path: Path | None = None,
) -> dict[str, Any]:
    """Measure a post-ending voice-band burst and retain frame-aligned ticks."""
    log_power, voice, seconds = voice_spectrogram(samples)
    candidate_seconds = (candidate_frame - 1.0) / fps
    base = {
        "schema": "tennis_ending_audio_call_v1",
        "candidate_frame": float(candidate_frame),
        "candidate_seconds": candidate_seconds,
        "post_window_seconds": list(CALL_POST_SECONDS),
        "voice_band_hz": list(VOICE_BAND_HZ),
        "out_call_likelihood_ratio": 1.0,
        "abstained": True,
        "abstention_reason": None,
    }
    if not len(voice):
        return {**base, "abstention_reason": "audio_window_too_short"}
    relative = seconds - candidate_seconds
    baseline_mask = (relative >= CALL_BASELINE_SECONDS[0]) & (relative <= CALL_BASELINE_SECONDS[1])
    post_mask = (relative >= CALL_POST_SECONDS[0]) & (relative <= CALL_POST_SECONDS[1])
    if baseline_mask.sum() < 5 or post_mask.sum() < 5:
        return {**base, "abstention_reason": "candidate_lacks_audio_context"}
    centre = float(np.median(voice[baseline_mask]))
    mad = float(np.median(np.abs(voice[baseline_mask] - centre)))
    robust_scale = max(1.4826 * mad, 0.05)
    z = (voice - centre) / robust_scale
    statistic = float(np.percentile(z[post_mask], 95))
    log_lr = (statistic - calibration.midpoint) / max(calibration.scale, 1e-6)
    likelihood_ratio = float(
        np.clip(
            math.exp(log_lr),
            1.0 / calibration.maximum_likelihood_ratio,
            calibration.maximum_likelihood_ratio,
        )
    )
    post_indices = np.flatnonzero(post_mask)
    receipt = {
        **base,
        "voice_burst_robust_z_p95": statistic,
        "baseline_log_energy_median": centre,
        "baseline_log_energy_mad": mad,
        "out_call_likelihood_ratio": likelihood_ratio,
        "aligned_frame_ticks": [
            {
                "audio_seconds": float(seconds[index]),
                "native_frame": float(seconds[index] * fps + 1.0),
                "voice_robust_z": float(z[index]),
            }
            for index in post_indices[:: max(1, len(post_indices) // 24)]
        ],
        "decoded_samples_sha256": hashlib.sha256(
            np.ascontiguousarray(samples).tobytes()
        ).hexdigest(),
        "calibration": asdict(calibration),
        "abstained": False,
        "abstention_reason": None,
    }
    if render_path is not None:
        render_spectrogram(log_power, seconds, fps, candidate_frame, render_path)
        receipt["spectrogram_strip"] = {
            "path": str(render_path),
            "sha256": hashlib.sha256(render_path.read_bytes()).hexdigest(),
        }
    return receipt


def render_spectrogram(
    log_power: np.ndarray,
    seconds: np.ndarray,
    fps: float,
    candidate_frame: float,
    path: Path,
) -> None:
    """Write a compact review strip; display pixels never feed the witness."""
    if not log_power.size:
        raise ValueError("cannot render an empty spectrogram")
    visible = log_power[:, : int(8_000 / (AUDIO_SAMPLE_RATE / AUDIO_FFT_SAMPLES)) + 1].T
    low, high = np.percentile(visible, [5, 99])
    scaled = np.uint8(np.clip((visible - low) * 255.0 / max(high - low, 1e-6), 0, 255))
    color = cv2.applyColorMap(np.flipud(scaled), cv2.COLORMAP_MAGMA)
    color = cv2.resize(color, (max(320, color.shape[1] * 3), 160), interpolation=cv2.INTER_NEAREST)
    candidate_seconds = (candidate_frame - 1.0) / fps
    x = int(round(np.interp(candidate_seconds, [seconds[0], seconds[-1]], [0, color.shape[1] - 1])))
    cv2.line(color, (x, 0), (x, color.shape[0] - 1), (255, 255, 255), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), color):
        raise OSError(f"could not write spectrogram {path}")


def shipped_ending_evidence(events: Sequence[dict]) -> dict[str, Any]:
    endings = [row for row in events if row.get("event_type") == "point_end"]
    if not endings:
        return {
            "schema": "tennis_shipped_ending_evidence_v1",
            "kind": None,
            "frame": None,
            "confidence": 0.0,
            "abstained": True,
            "abstention_reason": "no_shipped_point_end",
        }
    row = max(endings, key=lambda value: float(value.get("confidence", 0.0)))
    source_kind = str((row.get("point_end") or {}).get("termination_kind", ""))
    kind = _normalize_ending_kind(source_kind)
    return {
        "schema": "tennis_shipped_ending_evidence_v1",
        "kind": kind,
        "source_kind": source_kind,
        "frame": float(row["frame"]),
        "confidence": float(row.get("confidence", 0.0)),
        "abstained": kind is None,
        "abstention_reason": None if kind else "unsupported_shipped_ending_kind",
    }


def _normalize_ending_kind(value: str) -> str | None:
    value = value.lower()
    if "second" in value or "double" in value:
        return "second_bounce"
    if "out" in value or "wide" in value or "long" in value:
        return "out"
    if "net" in value:
        return "net"
    if "fov" in value or "view" in value:
        return "fov_exit"
    return None


def build_prior(
    segment: Segment,
    *,
    segment_index: int,
    adjacency: Mapping[str, Any],
    scoreboard: Mapping[str, Any],
    audio_call: Mapping[str, Any],
    shipped_ending: Mapping[str, Any],
    provenance_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Fuse independent receipts into conservative normalized relative weights."""
    serve = {key: 1.0 for key in ("first", "second", "fault", "let")}
    ending = {key: 1.0 for key in ("out", "second_bounce", "net", "fov_exit")}
    winner = {key: 0.0 for key in ("server", "receiver", "unknown")}
    winner["unknown"] = 1.0
    abstentions = []

    relation = adjacency.get("relation")
    if not adjacency.get("abstained") and relation == "same_server_same_side_fault_or_let":
        fault = float(adjacency.get("fault_likelihood_within_fault_or_let", 0.8))
        let = float(adjacency.get("let_likelihood_within_fault_or_let", 0.2))
        serve.update({"first": 0.20, "second": 0.05, "fault": 3.0 * fault, "let": 3.0 * let})
    elif not adjacency.get("abstained") and relation == "service_side_alternated_completed_point":
        ordinal = adjacency.get("completed_serve_ordinal")
        serve.update(
            {
                "first": 2.0 if ordinal == "first" else 0.4 if ordinal == "second" else 1.4,
                "second": 2.0 if ordinal == "second" else 0.4 if ordinal == "first" else 1.0,
                "fault": 0.05,
                "let": 0.05,
            }
        )
    else:
        abstentions.append("segment_adjacency")

    shipped_kind = shipped_ending.get("kind")
    if not shipped_ending.get("abstained") and shipped_kind in ending:
        confidence = float(shipped_ending.get("confidence", 0.0))
        for key in ending:
            ending[key] *= 1.0 + (8.0 if key == shipped_kind else -0.75) * confidence
    else:
        abstentions.append("shipped_ending")

    if not audio_call.get("abstained"):
        ratio = float(audio_call.get("out_call_likelihood_ratio", 1.0))
        ending["out"] *= math.sqrt(ratio)
        ending["second_bounce"] /= math.sqrt(ratio)
    else:
        abstentions.append("audio_call")

    if not scoreboard.get("abstained") and scoreboard.get("winner") in ("server", "receiver"):
        winner = {"server": 0.025, "receiver": 0.025, "unknown": 0.05}
        winner[str(scoreboard["winner"])] = 0.90
    else:
        abstentions.append("scoreboard_winner")
    winner = normalized_likelihoods(winner, ("server", "receiver", "unknown"))
    winner_label = max(winner, key=winner.get)
    if winner_label == "unknown" or winner[winner_label] < 0.80:
        winner_label = "unknown"

    document = {
        "schema": SCHEMA,
        "automatic": True,
        "match_id": segment.match_id,
        "point_id": segment.point_id,
        "segment_index": int(segment_index),
        "serve_ordinal": normalized_likelihoods(serve, ("first", "second", "fault", "let")),
        "ending_kind": normalized_likelihoods(ending, ("out", "second_bounce", "net", "fov_exit")),
        "winner": winner_label,
        "winner_likelihoods": winner,
        "evidence_receipts": {
            "segment_adjacency": dict(adjacency),
            "scoreboard": dict(scoreboard),
            "audio_call": dict(audio_call),
            "shipped_ending": dict(shipped_ending),
        },
        "abstentions": sorted(set(abstentions)),
        "provenance": dict(provenance_receipt),
    }
    validate_prior(document)
    return document


def validate_prior(document: Mapping[str, Any]) -> None:
    """Fail closed on a malformed or human-derived consumer artifact."""
    if document.get("schema") != SCHEMA or document.get("automatic") is not True:
        raise ValueError("unsupported or non-automatic point-context prior")
    if document.get("labels_or_reviewed_inputs"):
        raise ValueError("automatic point context cannot contain reviewed inputs")
    provenance_record = document.get("provenance") or {}
    if provenance_record.get("human_derived") is True or provenance_record.get(
        "human_derived_inputs"
    ):
        raise ValueError("automatic point context cannot contain human-derived inputs")
    for receipt_name, receipt in (document.get("evidence_receipts") or {}).items():
        if not isinstance(receipt, Mapping):
            raise ValueError(f"invalid {receipt_name} evidence receipt")
        if receipt.get("human_derived") is True or receipt.get("human_derived_inputs"):
            raise ValueError("automatic point-context receipt names human-derived input")
        if receipt.get("labels_or_reviewed_inputs"):
            raise ValueError("automatic point-context receipt names reviewed input")
    for field, keys in (
        ("serve_ordinal", ("first", "second", "fault", "let")),
        ("ending_kind", ("out", "second_bounce", "net", "fov_exit")),
        ("winner_likelihoods", ("server", "receiver", "unknown")),
    ):
        values = document.get(field)
        if not isinstance(values, Mapping) or set(values) != set(keys):
            raise ValueError(f"invalid {field} keys")
        numbers = np.asarray([values[key] for key in keys], dtype=float)
        if (
            not np.all(np.isfinite(numbers))
            or np.any(numbers < 0)
            or not np.isclose(numbers.sum(), 1.0, atol=1e-8)
        ):
            raise ValueError(f"invalid {field} likelihoods")


def normalize_consumer_ending_kind(value: str | None) -> str | None:
    """Map decoder/fitter ending names onto the point-context four-way contract."""

    normalized = (value or "").strip().lower().replace("-", "_")
    if "second" in normalized or "double" in normalized or "winner" in normalized:
        return "second_bounce"
    if "net" in normalized:
        return "net"
    if "out" in normalized or "wide" in normalized or "long" in normalized:
        return "out"
    if "fov" in normalized or "view" in normalized or "exit" in normalized:
        return "fov_exit"
    return None


def ending_witness(
    document: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Derive a soft physical-ending witness from validated source-only context.

    The first contact is the serve, so contact parity resolves whether the last
    hitter is the server or receiver even when the automatic side field is
    absent.  A known contact side is used as a cross-check and disagreement
    abstains from the winner-derived term.  No argmax is promoted to truth.
    """

    validate_prior(document)
    declared_end_frames = [
        float(row["frame"])
        for row in events
        if row.get("event_type") in {"ending", "point_end"} and row.get("frame") is not None
    ]
    declared_end = min(declared_end_frames, default=math.inf)
    contacts = sorted(
        (row for row in events if row.get("event_type") == "contact"),
        key=lambda row: float(row.get("frame", 0.0)),
    )
    contacts = [row for row in contacts if float(row.get("frame", 0.0)) <= declared_end]
    abstentions: list[str] = []
    last_hitter_role = None
    role_source = None
    if contacts:
        first_side = str(contacts[0].get("side") or contacts[0].get("hitter_end") or "")
        last_side = str(contacts[-1].get("side") or contacts[-1].get("hitter_end") or "")
        parity_role = "server" if len(contacts) % 2 else "receiver"
        if first_side in {"near", "far"} and last_side in {"near", "far"}:
            side_role = "server" if first_side == last_side else "receiver"
            if side_role == parity_role:
                last_hitter_role = side_role
                role_source = "contact_side_and_parity"
            else:
                abstentions.append("last_hitter_role_side_parity_disagreement")
        else:
            last_hitter_role = parity_role
            role_source = "serve_first_contact_parity"
    else:
        abstentions.append("no_contact_for_last_hitter_role")

    source = document["ending_kind"]
    likelihoods = {
        "first_bounce_out": float(source["out"]),
        "second_bounce": float(source["second_bounce"]),
        "net_hit": float(source["net"]),
        "fov_exit": float(source["fov_exit"]),
    }
    winner = document["winner_likelihoods"]
    last_ball_in = 0.5
    winner_used = False
    if last_hitter_role in {"server", "receiver"}:
        other = "receiver" if last_hitter_role == "server" else "server"
        known_mass = float(winner[last_hitter_role]) + float(winner[other])
        if known_mass > 0.5:
            # Score tells us who won, not whether the losing stroke missed the
            # court or the net. Keep both explanations alive and bounded.
            last_ball_in = (
                0.85 * float(winner[last_hitter_role])
                + 0.15 * float(winner[other])
                + 0.50 * float(winner["unknown"])
            )
            in_ratio = max(0.25, min(4.0, last_ball_in / 0.5))
            error_ratio = max(0.25, min(4.0, (1.0 - last_ball_in) / 0.5))
            likelihoods["second_bounce"] *= math.sqrt(in_ratio)
            likelihoods["fov_exit"] *= math.sqrt(in_ratio)
            likelihoods["first_bounce_out"] *= math.sqrt(error_ratio)
            likelihoods["net_hit"] *= math.sqrt(error_ratio)
            winner_used = True
        else:
            abstentions.append("scoreboard_winner")
    else:
        abstentions.append("scoreboard_winner_without_last_hitter_role")

    serve = document["serve_ordinal"]
    fault_or_let = float(serve["fault"]) + float(serve["let"])
    serve_restart_used = bool(len(contacts) == 1 and fault_or_let > 0.5)
    if serve_restart_used:
        ratio = max(0.25, min(4.0, fault_or_let / 0.5))
        likelihoods["first_bounce_out"] *= math.sqrt(ratio)
        likelihoods["net_hit"] *= math.sqrt(ratio)
        likelihoods["second_bounce"] /= math.sqrt(ratio)
    elif len(contacts) == 1:
        abstentions.append("same_side_serve_restart")

    normalized = normalized_likelihoods(
        likelihoods, ("first_bounce_out", "second_bounce", "net_hit", "fov_exit")
    )
    point_ended = normalized_likelihoods(
        {
            "first_bounce": normalized["first_bounce_out"],
            "second_bounce": normalized["second_bounce"],
            "net": normalized["net_hit"],
        },
        ("first_bounce", "second_bounce", "net"),
    )
    return {
        "schema": "tennis_ending_context_witness_v1",
        "automatic": True,
        "available": bool(
            winner_used or serve_restart_used or "audio_call" not in document["abstentions"]
        ),
        "ending_likelihoods": normalized,
        "last_ball_in_likelihoods": {
            "in": float(last_ball_in),
            "out_or_net": float(1.0 - last_ball_in),
        },
        "point_ended_at_this_bounce_likelihoods": point_ended,
        "last_hitter_role": last_hitter_role,
        "last_hitter_role_source": role_source,
        "scoreboard_winner_used": winner_used,
        "same_side_serve_restart_used": serve_restart_used,
        "audio_out_call_used": "audio_call" not in document["abstentions"],
        "abstentions": sorted(set(abstentions)),
        "source_context": {
            "schema": document["schema"],
            "match_id": document.get("match_id"),
            "point_id": document.get("point_id"),
        },
        "hard_gate": False,
    }


def _load_player_rows(match_dir: Path) -> list[dict[str, str]]:
    candidates = sorted(match_dir.glob("player_boxes_*_native_sided_v1.csv"))
    if not candidates:
        candidates = sorted(match_dir.glob("player_tracks_native_v1.csv"))
    if not candidates:
        return []
    # The box columns are read in whatever space the sidecar declares; a file name is
    # never coordinate evidence, and a file without a sidecar is not read at all.
    try:
        scale_x, scale_y = res.coordinate_scale(candidates[0], columns=("x0", "y0", "x1", "y1"))
    except res.MissingCoordinateContract:
        return []
    with candidates[0].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["_scale_x"] = str(scale_x)
        row["_scale_y"] = str(scale_y)
    return rows


def _score_at_segment_start(
    match_dir: Path,
    segment: Segment,
    score_states: Sequence[tuple[float, serve_speed_graphic.ScoreOverlayReading]] = (),
) -> serve_speed_graphic.ScoreOverlayReading:
    if (
        score_states
        and segment.source_timeline_available
        and segment.source_start_seconds is not None
    ):
        source_seconds, reading = min(
            score_states, key=lambda item: abs(item[0] - float(segment.source_start_seconds))
        )
        if abs(source_seconds - float(segment.source_start_seconds)) <= 0.51:
            return replace(
                reading,
                frame=max(1, int(round(segment.source_start_seconds * segment.fps)) + 1),
            )
    if segment.source_timeline_available and segment.source_start_seconds is not None:
        source = match_dir / "audit_reel_native_1080.mp4"
        capture = cv2.VideoCapture(str(source))
        capture.set(cv2.CAP_PROP_POS_MSEC, 1_000.0 * segment.source_start_seconds)
        okay, image = capture.read()
        capture.release()
        source_frame = max(1, int(round(segment.source_start_seconds * segment.fps)) + 1)
        if okay and image is not None:
            return serve_speed_graphic.read_score_overlay(image, source_frame)
        return serve_speed_graphic.ScoreOverlayReading(
            source_frame,
            False,
            (),
            None,
            0.0,
            True,
            "native_source_score_frame_unreadable",
            None,
            None,
            "",
            (),
        )
    frame = max(1, int(round(segment.start_frame)))
    path = match_dir / "audit_frames_native_1080" / segment.point_id / f"f_{frame:04d}.jpg"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return serve_speed_graphic.ScoreOverlayReading(
            frame, False, (), None, 0.0, True, "native_score_frame_unreadable", None, None, "", ()
        )
    return serve_speed_graphic.read_score_overlay(image, frame)


def load_score_state_ledger(
    path: Path,
) -> list[tuple[float, serve_speed_graphic.ScoreOverlayReading]]:
    """Load one explicitly named automatic score-grammar state ledger."""

    output = []
    ledger_sha256 = str(_cached_file_record(str(path.resolve()))["sha256"])
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                source_seconds = float(row["t"])
                server = int(row["server"])
                games = (str(int(float(row["games_1"]))), str(int(float(row["games_2"]))))
                points = (str(row["points_1"]), str(row["points_2"]))
                confidence = float(row["score_confidence"])
            except (KeyError, TypeError, ValueError):
                continue
            if server not in (1, 2) or not all(points) or not math.isfinite(confidence):
                continue
            output.append(
                (
                    source_seconds,
                    serve_speed_graphic.ScoreOverlayReading(
                        frame=0,
                        present=True,
                        rows=(
                            {"name": "PLAYER1", "values": [games[0]], "points": points[0]},
                            {"name": "PLAYER2", "values": [games[1]], "points": points[1]},
                        ),
                        serving_row=server,
                        confidence=confidence,
                        abstained=False,
                        abstention_reason=None,
                        roi_native_xyxy=None,
                        crop_sha256=ledger_sha256,
                        raw_text=f"automatic_score_state_v1 t={source_seconds:.3f}",
                        words=(),
                    ),
                )
            )
    return output


def _candidate_frame(events: Sequence[dict], segment: Segment) -> tuple[float, str]:
    endings = [row for row in events if row.get("event_type") == "point_end"]
    if endings:
        # Keep witness placement consistent with shipped_ending_evidence: a
        # lower-confidence late alternative must not move the audio window
        # beyond the selected physical ending.
        selected = max(
            endings,
            key=lambda row: (float(row.get("confidence", 0.0)), -float(row["frame"])),
        )
        return float(selected["frame"]), "shipped_point_end"
    physical = [row for row in events if row.get("event_type") in {"bounce", "net_hit", "contact"}]
    if physical:
        return float(
            max(physical, key=lambda row: float(row["frame"]))["frame"]
        ), "last_emitted_physical_event"
    return float(segment.end_frame), "active_play_end_fallback"


def _tool_identity(executable: str, version_flag: str) -> dict[str, Any]:
    """Bind an external executable path and its first version line."""
    resolved = shutil.which(executable)
    if resolved is None:
        return {"path": None, "version": None, "available": False}
    try:
        result = subprocess.run(
            [resolved, version_flag], capture_output=True, check=False, text=True, timeout=10
        )
        output = result.stdout or result.stderr
        version = output.splitlines()[0].strip() if output else None
    except (OSError, subprocess.SubprocessError):
        version = None
    return {"path": resolved, "version": version, "available": True}


def _audio_for_segment(
    match_dir: Path,
    segment: Segment,
    candidate_frame: float,
    render_path: Path,
    calibration: CallCalibration,
) -> dict[str, Any]:
    if segment.source_timeline_available and segment.source_start_seconds is not None:
        source = match_dir / "audit_reel_native_1080.mp4"
        source_candidate_seconds = (
            segment.source_start_seconds + (candidate_frame - segment.start_frame) / segment.fps
        )
        try:
            return source_audio_call_witness(
                source,
                source_candidate_seconds=source_candidate_seconds,
                fps=segment.fps,
                render_path=render_path,
                calibration=calibration,
            )
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            return {
                "schema": "tennis_ending_audio_call_v1",
                "abstained": True,
                "abstention_reason": f"source_audio_decode_failed:{type(error).__name__}",
                "out_call_likelihood_ratio": 1.0,
            }
    cache = match_dir / "contact_audio_scores_16k_native_v1.npz"
    if not cache.is_file():
        return {
            "schema": "tennis_ending_audio_call_v1",
            "abstained": True,
            "abstention_reason": "aligned_audio_cache_unavailable",
            "out_call_likelihood_ratio": 1.0,
        }
    try:
        with np.load(cache, allow_pickle=False) as stored:
            metadata = json.loads(str(stored["metadata"].item()))
        window = metadata["windows"][segment.point_id]
        video = Path(metadata["video"])
        samples = decode_audio(video, float(window[0]), float(window[1]) - float(window[0]))
        receipt = audio_call_witness(
            samples,
            fps=segment.fps,
            candidate_frame=candidate_frame,
            calibration=calibration,
            render_path=render_path,
        )
        return {
            **receipt,
            "audio_cache": provenance.file_record(cache),
            "source_video": provenance.file_record(video),
            "clip_audio_window_seconds": [float(window[0]), float(window[1])],
        }
    except (
        OSError,
        KeyError,
        ValueError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
    ) as error:
        return {
            "schema": "tennis_ending_audio_call_v1",
            "abstained": True,
            "abstention_reason": f"audio_decode_failed:{type(error).__name__}",
            "out_call_likelihood_ratio": 1.0,
        }


def source_audio_call_witness(
    video: Path,
    *,
    source_candidate_seconds: float,
    fps: float,
    render_path: Path | None = None,
    calibration: CallCalibration = CallCalibration(),
) -> dict[str, Any]:
    """Run the ending call witness around an absolute source timestamp."""
    window_start = max(0.0, source_candidate_seconds + CALL_BASELINE_SECONDS[0] - 0.1)
    window_end = source_candidate_seconds + CALL_POST_SECONDS[1] + 0.1
    samples = decode_audio(video, window_start, window_end - window_start)
    local_candidate = (source_candidate_seconds - window_start) * fps + 1.0
    receipt = audio_call_witness(
        samples,
        fps=fps,
        candidate_frame=local_candidate,
        calibration=calibration,
        render_path=render_path,
    )
    ticks = []
    for tick in receipt.get("aligned_frame_ticks", []):
        source_seconds = window_start + float(tick["audio_seconds"])
        ticks.append(
            {
                **tick,
                "audio_seconds": source_seconds,
                "native_frame": source_seconds * fps + 1.0,
            }
        )
    return {
        **receipt,
        "candidate_frame": source_candidate_seconds * fps + 1.0,
        "candidate_seconds": source_candidate_seconds,
        "aligned_frame_ticks": ticks,
        "source_audio_window_seconds": [window_start, window_end],
        "source_video": _cached_file_record(str(video.resolve())),
    }


def run_cohort(
    *,
    active_play_path: Path,
    artifact_roots: Sequence[Path],
    emissions_path: Path,
    output_dir: Path,
    source_timeline: bool | None = None,
    calibration: CallCalibration = CallCalibration(),
    score_state_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    """Materialize one automatic JSON sidecar per available cohort point."""
    if output_dir.exists():
        raise ValueError("point-context output directory must be new")
    segments = load_active_segments(active_play_path)
    emissions = load_emissions(emissions_path)
    root_by_match = {
        child.name: child
        for root in artifact_roots
        if root.is_dir()
        for child in root.iterdir()
        if child.is_dir()
    }
    score_state_path_by_match = {path.parent.name: path for path in score_state_paths}
    score_states_by_match = {
        match_id: load_score_state_ledger(path)
        for match_id, path in score_state_path_by_match.items()
    }
    hydrated = []
    for match_id in sorted({segment.match_id for segment in segments}):
        match_segments = [segment for segment in segments if segment.match_id == match_id]
        match_dir = root_by_match.get(match_id)
        hydrated.extend(
            _source_timeline_segments(match_segments, match_dir, source_timeline)
            if match_dir
            else match_segments
        )
    segments = sorted(hydrated, key=lambda row: (row.match_id, _point_number(row.point_id)))
    neighbors = adjacency_windows(segments, radius=3)
    servers = {}
    scores = {}
    rows_by_match: dict[str, list[Segment]] = {}
    for segment in segments:
        rows_by_match.setdefault(segment.match_id, []).append(segment)
        match_dir = root_by_match.get(segment.match_id)
        events = emissions.get((segment.match_id, segment.point_id), [])
        player_rows = _load_player_rows(match_dir) if match_dir else []
        servers[(segment.match_id, segment.point_id)] = infer_server_position(
            segment, events, player_rows
        )
        scores[(segment.match_id, segment.point_id)] = (
            _score_at_segment_start(
                match_dir, segment, score_states_by_match.get(segment.match_id, ())
            )
            if match_dir
            else serve_speed_graphic.ScoreOverlayReading(
                int(segment.start_frame),
                False,
                (),
                None,
                0.0,
                True,
                "match_artifact_directory_unavailable",
                None,
                None,
                "",
                (),
            )
        )

    output_dir.mkdir(parents=True)
    points = []
    source_videos: dict[str, dict[str, Any]] = {}
    fallback_counts: Counter[str] = Counter()
    for match_id, match_segments in sorted(rows_by_match.items()):
        match_segments.sort(key=lambda row: _point_number(row.point_id))
        match_dir = root_by_match.get(match_id)
        for index, segment in enumerate(match_segments):
            following = match_segments[index + 1] if index + 1 < len(match_segments) else None
            current_events = emissions.get((match_id, segment.point_id), [])
            adjacency = segment_adjacency_evidence(
                segment,
                following,
                servers[(match_id, segment.point_id)],
                servers.get((match_id, following.point_id)) if following else None,
                current_events,
            )
            adjacency["neighbors"] = neighbors[(match_id, segment.point_id)]
            score_following = [
                (candidate, scores[(match_id, candidate.point_id)])
                for candidate in match_segments[index + 1 : index + 4]
            ]
            score = scoreboard_lookahead_evidence(
                scores[(match_id, segment.point_id)], score_following
            )
            candidate_frame, candidate_source = _candidate_frame(current_events, segment)
            audio = (
                _audio_for_segment(
                    match_dir,
                    segment,
                    candidate_frame,
                    output_dir / "spectrograms" / match_id / f"{segment.point_id}.png",
                    calibration,
                )
                if match_dir
                else {
                    "schema": "tennis_ending_audio_call_v1",
                    "abstained": True,
                    "abstention_reason": "match_artifact_directory_unavailable",
                    "out_call_likelihood_ratio": 1.0,
                }
            )
            audio["candidate_source"] = candidate_source
            shipped = shipped_ending_evidence(current_events)
            context = build_prior(
                segment,
                segment_index=index,
                adjacency=adjacency,
                scoreboard=score,
                audio_call=audio,
                shipped_ending=shipped,
                provenance_receipt={
                    "active_play": provenance.file_record(active_play_path),
                    "emissions": provenance.file_record(emissions_path),
                    "match_artifact_directory": str(match_dir) if match_dir else None,
                    "implementation": provenance.file_record(Path(__file__)),
                    "score_reader_implementation": provenance.file_record(
                        Path(serve_speed_graphic.__file__)
                    ),
                    "score_state_ledger": (
                        provenance.file_record(score_state_path_by_match[match_id])
                        if match_id in score_state_path_by_match
                        else None
                    ),
                    "code": provenance.git_record(paths.REPO_ROOT),
                    "human_derived_inputs": [],
                    "native_timestamps_changed": False,
                },
            )
            target = output_dir / "points" / match_id / f"{segment.point_id}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(context, indent=2, sort_keys=True) + "\n")
            points.append({"match_id": match_id, "point_id": segment.point_id, "path": str(target)})
            if audio.get("source_video"):
                source_videos[str(audio["source_video"]["path"])] = audio["source_video"]
            for receipt_name, receipt in context["evidence_receipts"].items():
                if receipt.get("abstained"):
                    reason = receipt.get("abstention_reason") or "unspecified"
                    fallback_counts[f"{receipt_name}:{reason}"] += 1
    manifest = {
        "schema": "tennis_point_context_run_v1",
        "automatic": True,
        "status": "complete",
        "points": points,
        "point_count": len(points),
        "broadcast_count": len(rows_by_match),
        "source_timeline_available": any(row.source_timeline_available for row in segments),
        "timeline_mode": (
            "source"
            if segments and all(row.source_timeline_available for row in segments)
            else "mixed"
            if any(row.source_timeline_available for row in segments)
            else "reel_fallback"
        ),
        "configuration": {
            "adjacency_radius": 3,
            "call_calibration": asdict(calibration),
            "source_timeline_requested": source_timeline,
        },
        "inputs": {
            "active_play": provenance.file_record(active_play_path),
            "emissions": provenance.file_record(emissions_path),
            "reused_artifact_roots": [str(path) for path in artifact_roots],
            "source_videos": sorted(source_videos.values(), key=lambda row: str(row["path"])),
            "score_state_ledgers": [
                provenance.file_record(path, role="automatic_score_state")
                for path in score_state_paths
            ],
        },
        "implementations": {
            "point_context": provenance.file_record(Path(__file__)),
            "score_reader": provenance.file_record(Path(serve_speed_graphic.__file__)),
        },
        "external_tools": {
            "ffmpeg": _tool_identity("ffmpeg", "-version"),
            "tesseract": _tool_identity("tesseract", "--version"),
        },
        "code": provenance.git_record(paths.REPO_ROOT),
        "fallback_counts": dict(sorted(fallback_counts.items())),
        "human_derived_inputs": [],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-play", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, action="append", required=True)
    parser.add_argument("--emissions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--score-state-ledger",
        type=Path,
        action="append",
        default=[],
        help=(
            "explicit automatic score_state_v1.csv; repeat per broadcast. "
            "Its parent directory name must be the match id"
        ),
    )
    timeline = parser.add_mutually_exclusive_group()
    timeline.add_argument(
        "--source-timeline",
        dest="source_timeline",
        action="store_true",
        help="require source timing where the source-backed point map is available",
    )
    timeline.add_argument(
        "--reel-timeline",
        dest="source_timeline",
        action="store_false",
        help="force legacy concatenated audit-reel timing",
    )
    parser.set_defaults(source_timeline=None)
    parser.add_argument("--call-midpoint", type=float, default=CallCalibration.midpoint)
    parser.add_argument("--call-scale", type=float, default=CallCalibration.scale)
    args = parser.parse_args()
    manifest = run_cohort(
        active_play_path=args.active_play,
        artifact_roots=args.artifact_root,
        emissions_path=args.emissions,
        output_dir=args.output_dir,
        source_timeline=args.source_timeline,
        calibration=CallCalibration(midpoint=args.call_midpoint, scale=args.call_scale),
        score_state_paths=args.score_state_ledger,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
