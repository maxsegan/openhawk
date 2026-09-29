"""Serve-anchored attempt segmentation for Stage 1.

One physical attempt starts at a serve.  A first-serve fault and the second serve that
follows it are two attempts inside one continuous play-camera shot, which is exactly the case
the old audio-gap splitter could not see: racket-impact peaks fire far too often to mark the
boundary, so faults and lets were merged into their point.

This module scores every 0.2 s of play-camera time with two small gradient-boosted heads over
a shared temporal feature window:

* ``start`` - the ideal Stage-1 segment start (a serve about to happen);
* ``end``   - the ideal Stage-1 segment end (the attempt is over).

Features are the native-cadence motion energy from ``shot_boundaries``, the racket-impact
audio flux from ``serve_audio``, the shot/view structure from ``view_classifier``, native
player boxes (how many people, how tall, how still, how far apart vertically), and the
distance to the nearest scoreboard change.  Attempts are then assembled greedily: each
accepted start opens an attempt, which closes at the best end peak before the next start, the
end of its shot, or a score change - whichever comes first.

Heads are fitted on the development windows only and evaluated leave-one-broadcast-out, so no
reported number is scored by a model that saw that broadcast.

    .venv/bin/python -m cv.pipeline.serve_detector predict --out out_dir --model serve.json
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

from cv.pipeline.artifact_cache import stage_receipt_matches, write_stage_receipt

GRID_SECONDS = 0.2
# The closest two owner-labelled attempt starts in the twelve development windows are 6.01 s
# apart, and none is closer; two accepted starts inside that cannot both be serves.
MINIMUM_START_GAP_SECONDS = 5.5
MINIMUM_ATTEMPT_SECONDS = 1.0
MAXIMUM_ATTEMPT_SECONDS = 45.0
# Owner segment ends sit a little after the terminal event, to keep post-point audio and
# context; the median gap from the last racket impact to the owner end on the development
# windows is measured in `cv.validation.s1_stage_v2_cohort` and defaults to this.
LAST_IMPACT_BUFFER_SECONDS = 1.6
# Serves arrive roughly every twenty seconds; a much longer hole inside continuous play-camera
# time is a missed serve, not a quiet patch of tennis.
GAP_FILL_SECONDS = 34.0
GAP_FILL_THRESHOLD = 0.25


# ---------------- native player boxes ------------------------------------------------------


def player_boxes(
    frame_paths: list[Path],
    *,
    weights: Path,
    device: str = "0",
    imgsz: int = 1920,
    batch: int = 16,
) -> np.ndarray:
    """Per-frame person-box summary from native 1920x1080 frames.

    Columns: people, tallest height fraction, second height fraction, lowest box bottom,
    highest box bottom, mean box centre x spread, tallest box centre y.
    """
    from ultralytics import YOLO

    model = YOLO(os.fspath(weights))
    rows = np.zeros((len(frame_paths), 7), dtype=np.float32)
    for start in range(0, len(frame_paths), batch):
        chunk = [os.fspath(path) for path in frame_paths[start : start + batch]]
        predictions = model.predict(
            chunk, classes=[0], imgsz=imgsz, device=device, verbose=False, half=True
        )
        for offset, prediction in enumerate(predictions):
            boxes = prediction.boxes.xyxy.cpu().numpy()
            height, width = prediction.orig_shape
            if not len(boxes):
                continue
            heights = (boxes[:, 3] - boxes[:, 1]) / height
            order = np.argsort(-heights)
            bottoms = boxes[:, 3] / height
            centres = 0.5 * (boxes[:, 0] + boxes[:, 2]) / width
            rows[start + offset] = (
                len(boxes),
                heights[order[0]],
                heights[order[1]] if len(boxes) > 1 else 0.0,
                bottoms.max(),
                bottoms.min(),
                float(centres.max() - centres.min()),
                float(0.5 * (boxes[order[0], 1] + boxes[order[0], 3]) / height),
            )
    return rows


PLAYER_COLUMNS = (
    "people",
    "tallest_height",
    "second_height",
    "lowest_bottom",
    "highest_bottom",
    "centre_spread",
    "tallest_centre_y",
)


# ---------------- temporal features --------------------------------------------------------


def _uniform_window(values: np.ndarray, fps: float, offset: float, t0: float, t1: float):
    low = max(0, int(np.floor((t0 - offset) * fps)))
    high = min(len(values), int(np.ceil((t1 - offset) * fps)))
    return values[low:high] if high > low else values[low : low + 1]


def _stat(values: np.ndarray, kind: str) -> float:
    if not len(values):
        return 0.0
    if kind == "mean":
        return float(values.mean())
    if kind == "max":
        return float(values.max())
    return float(np.median(values))


AUDIO_WINDOWS = ((-0.4, 0.4), (-1.5, 0.0), (0.0, 1.5), (-4.0, 0.0), (0.0, 4.0))
MOTION_WINDOWS = ((-2.0, -0.5), (-0.5, 0.5), (0.5, 2.0), (2.0, 5.0), (-5.0, -2.0))


def feature_names() -> list[str]:
    names = []
    for index, _ in enumerate(AUDIO_WINDOWS):
        names += [f"audio_max_{index}", f"audio_count_{index}"]
    names += ["audio_gap_before", "audio_gap_after"]
    for index, _ in enumerate(MOTION_WINDOWS):
        names += [f"motion_mean_{index}", f"motion_top_{index}", f"motion_bottom_{index}"]
    names += ["motion_ratio", "motion_top_ratio", "duplicate_fraction_local"]
    names += ["shot_elapsed", "shot_remaining", "shot_length", "view_play", "view_replay"]
    names += [f"player_{name}" for name in PLAYER_COLUMNS]
    names += ["player_stillness", "player_bottom_drift"]
    names += ["score_gap_before", "score_gap_after"]
    names += TERMINAL_FEATURE_NAMES
    names += PLAY_NORMALISED_FEATURE_NAMES
    names += TOSS_FEATURE_NAMES
    return names


# A serve is the one moment in tennis where the ball rises out of the server's hands, hangs,
# and is struck at the top of its arc.  Running the ball tracker over the whole broadcast to
# see that costs more than the rest of Stage 1 put together, so it runs only in a short window
# around each candidate start and the descriptors below are attached to the grid.
TOSS_FEATURE_NAMES = [
    "toss_present",
    "toss_offset",
    "toss_detection_rate",
    "toss_rise_fraction",
    "toss_apex_offset",
    "toss_arc_shape",
    "toss_strike_speed",
    "toss_apex_height",
]
NO_TOSS = [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# Racket-impact flux and motion energy carry a per-broadcast scale: the 2015 grass window fires
# 412 onsets a minute above the absolute z threshold where a 2024 window fires 1103, and their
# median motion differs threefold.  Pass one tried z-scoring every feature over the whole span
# and lost recall, because the whole span mixes play with crowd and close-up shots.  These few
# columns instead divide the scale-sensitive quantities by a reference measured on
# play-camera time only, and are added alongside the absolute ones rather than replacing them.
PLAY_NORMALISED_FEATURE_NAMES = [
    "norm_audio_max_centre",
    "norm_audio_max_before",
    "norm_audio_max_after",
    "norm_audio_count_centre",
    "norm_motion_before",
    "norm_motion_centre",
    "norm_motion_after",
    "norm_audio_next_3s",
]


# A rally ends at a terminal event, not at the next structural boundary.  The evidence is the
# last racket impact followed by audio that decays into applause, the players letting their
# motion fall away, and the scoreboard advancing a moment later.  These are computed looking
# forward from the moment, which the symmetric windows above deliberately are not.
TERMINAL_FEATURE_NAMES = [
    "terminal_onsets_next_1s",
    "terminal_onsets_next_3s",
    "terminal_onsets_next_6s",
    "terminal_last_onset_age",
    "terminal_audio_decay_1s",
    "terminal_audio_decay_3s",
    "terminal_audio_mean_next_3s",
    "terminal_audio_mean_next_6s",
    "terminal_motion_decay_2s",
    "terminal_player_stillness_after",
    "terminal_player_stillness_ratio",
    "terminal_score_change_2s",
    "terminal_score_change_5s",
    "terminal_score_change_10s",
]


def build_features(
    times: np.ndarray,
    *,
    signals: dict,
    audio_z: np.ndarray,
    audio_hop: float,
    audio_offset: float,
    shots: list[dict],
    view_probabilities: dict[int, dict[str, float]],
    player_rows: np.ndarray,
    player_times: np.ndarray,
    score_changes: np.ndarray,
    audio_reference: float = 1.0,
    motion_reference: float = 1.0,
    toss_times: np.ndarray | None = None,
    toss_values: np.ndarray | None = None,
) -> np.ndarray:
    fps = float(signals["fps"])
    offset = float(signals["start_seconds"])
    motion = signals["motion"]
    motion_top = signals["motion_top"]
    motion_bottom = signals["motion_bottom"]
    audio_fps = 1.0 / audio_hop
    onsets = audio_offset + np.where(audio_z >= 5.0)[0] * audio_hop if len(audio_z) else np.empty(0)
    shot_starts = np.array([float(shot["t_start"]) for shot in shots])
    shot_ends = np.array([float(shot["t_end"]) for shot in shots])
    audio_scale = max(1e-3, float(audio_reference))
    motion_scale = max(1e-3, float(motion_reference))
    rows = np.zeros((len(times), len(feature_names())), dtype=np.float32)
    for index, moment in enumerate(times):
        values = []
        for low, high in AUDIO_WINDOWS:
            window = _uniform_window(audio_z, audio_fps, audio_offset, moment + low, moment + high)
            values += [_stat(window, "max"), float((window >= 5.0).sum())]
        before = onsets[onsets <= moment]
        after = onsets[onsets > moment]
        values += [
            float(moment - before[-1]) if len(before) else 30.0,
            float(after[0] - moment) if len(after) else 30.0,
        ]
        for low, high in MOTION_WINDOWS:
            values += [
                _stat(_uniform_window(motion, fps, offset, moment + low, moment + high), "mean"),
                _stat(
                    _uniform_window(motion_top, fps, offset, moment + low, moment + high), "mean"
                ),
                _stat(
                    _uniform_window(motion_bottom, fps, offset, moment + low, moment + high),
                    "mean",
                ),
            ]
        pre = _stat(_uniform_window(motion, fps, offset, moment - 2.0, moment), "mean")
        post = _stat(_uniform_window(motion, fps, offset, moment, moment + 2.0), "mean")
        pre_top = _stat(_uniform_window(motion_top, fps, offset, moment - 2.0, moment), "mean")
        post_top = _stat(_uniform_window(motion_top, fps, offset, moment, moment + 2.0), "mean")
        local = _uniform_window(motion, fps, offset, moment - 1.0, moment + 1.0)
        values += [
            float(np.log1p(post) - np.log1p(pre)),
            float(np.log1p(post_top) - np.log1p(pre_top)),
            float((local < 0.06).mean()) if len(local) else 0.0,
        ]
        shot_index = int(np.searchsorted(shot_starts, moment, side="right") - 1)
        shot_index = min(max(shot_index, 0), len(shots) - 1)
        probabilities = view_probabilities.get(shot_index, {})
        values += [
            float(moment - shot_starts[shot_index]),
            float(shot_ends[shot_index] - moment),
            float(shot_ends[shot_index] - shot_starts[shot_index]),
            float(probabilities.get("play", 0.0)),
            float(probabilities.get("replay", 0.0)),
        ]
        nearest = int(np.argmin(np.abs(player_times - moment))) if len(player_times) else -1
        stillness_before = 0.0
        stillness_after = 0.0
        if nearest >= 0:
            values += list(player_rows[nearest])
            low = np.searchsorted(player_times, moment - 2.0)
            high = np.searchsorted(player_times, moment + 0.5)
            span = player_rows[low:high]
            if len(span) > 1:
                stillness_before = float(np.abs(np.diff(span[:, 6])).mean())
                values += [stillness_before, float(np.abs(np.diff(span[:, 3])).mean())]
            else:
                values += [0.0, 0.0]
            ahead = player_rows[
                np.searchsorted(player_times, moment) : np.searchsorted(player_times, moment + 2.5)
            ]
            if len(ahead) > 1:
                stillness_after = float(np.abs(np.diff(ahead[:, 6])).mean())
        else:
            values += [0.0] * (len(PLAYER_COLUMNS) + 2)
        before_score = score_changes[score_changes <= moment]
        after_score = score_changes[score_changes > moment]
        score_ahead = float(after_score[0] - moment) if len(after_score) else 120.0
        values += [
            float(moment - before_score[-1]) if len(before_score) else 120.0,
            score_ahead,
        ]
        # terminal-event evidence, all looking forward from the moment
        after_onsets = onsets[onsets > moment]
        values += [
            float(((after_onsets - moment) <= 1.0).sum()),
            float(((after_onsets - moment) <= 3.0).sum()),
            float(((after_onsets - moment) <= 6.0).sum()),
            float(moment - before[-1]) if len(before) else 30.0,
        ]
        back_1 = _stat(
            _uniform_window(audio_z, audio_fps, audio_offset, moment - 1.0, moment), "mean"
        )
        fwd_1 = _stat(
            _uniform_window(audio_z, audio_fps, audio_offset, moment, moment + 1.0), "mean"
        )
        back_3 = _stat(
            _uniform_window(audio_z, audio_fps, audio_offset, moment - 3.0, moment), "mean"
        )
        fwd_3 = _stat(
            _uniform_window(audio_z, audio_fps, audio_offset, moment, moment + 3.0), "mean"
        )
        fwd_6 = _stat(
            _uniform_window(audio_z, audio_fps, audio_offset, moment, moment + 6.0), "mean"
        )
        values += [fwd_1 - back_1, fwd_3 - back_3, fwd_3, fwd_6]
        motion_back = _stat(_uniform_window(motion, fps, offset, moment - 2.0, moment), "mean")
        motion_fwd = _stat(_uniform_window(motion, fps, offset, moment, moment + 2.0), "mean")
        values.append(float(np.log1p(motion_fwd) - np.log1p(motion_back)))
        values += [
            stillness_after,
            float(np.log1p(stillness_after) - np.log1p(stillness_before)),
        ]
        values += [
            float(score_ahead <= 2.0),
            float(score_ahead <= 5.0),
            float(score_ahead <= 10.0),
        ]
        centre = _uniform_window(audio_z, audio_fps, audio_offset, moment - 0.4, moment + 0.4)
        values += [
            _stat(centre, "max") / audio_scale,
            values[2] / audio_scale,
            values[4] / audio_scale,
            float((centre >= audio_reference).sum()),
            values[len(AUDIO_WINDOWS) * 2 + 2] / motion_scale,
            values[len(AUDIO_WINDOWS) * 2 + 5] / motion_scale,
            values[len(AUDIO_WINDOWS) * 2 + 8] / motion_scale,
            fwd_3 / audio_scale,
        ]
        if toss_times is not None and len(toss_times):
            nearest_toss = int(np.argmin(np.abs(toss_times - moment)))
            offset = float(toss_times[nearest_toss] - moment)
            if abs(offset) <= 2.5:
                descriptor = list(toss_values[nearest_toss])
                descriptor[1] = offset
                values += descriptor
            else:
                values += list(NO_TOSS)
        else:
            values += list(NO_TOSS)
        rows[index] = values
    return rows


def grid_times(start: float, end: float) -> np.ndarray:
    count = int(np.floor((end - start) / GRID_SECONDS)) + 1
    return start + np.arange(count) * GRID_SECONDS


# ---------------- heads --------------------------------------------------------------------


def fit_head(features: np.ndarray, labels: np.ndarray, *, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    positive = max(int(labels.sum()), 1)
    weight = len(labels) / (2.0 * positive)
    sample_weight = np.where(labels > 0, weight, 1.0)
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(features, labels, sample_weight=sample_weight)
    return model


def soft_labels(times: np.ndarray, anchors: np.ndarray, tolerance: float) -> np.ndarray:
    labels = np.zeros(len(times), dtype=np.int8)
    for anchor in anchors:
        near = np.where(np.abs(times - anchor) <= tolerance)[0]
        labels[near] = 1
    return labels


def peaks(
    scores: np.ndarray, times: np.ndarray, *, threshold: float, minimum_gap: float
) -> list[int]:
    # Equal-score plateaus must not change winners when unrelated source rows
    # are prepended. Prefer the earliest source epoch, then the original row.
    order = np.lexsort((np.arange(len(scores)), times, -scores))
    chosen: list[int] = []
    for index in order:
        if scores[index] < threshold:
            break
        if any(abs(times[index] - times[other]) < minimum_gap for other in chosen):
            continue
        chosen.append(int(index))
    return sorted(chosen)


def fill_start_gaps(
    starts: list[int],
    masked_scores: np.ndarray,
    times: np.ndarray,
    play_mask: np.ndarray,
    *,
    onsets: np.ndarray,
    gap_seconds: float,
    threshold: float,
    onset_tolerance: float = 1.0,
) -> list[int]:
    """Recover serves the head scored below threshold inside long holes in continuous play.

    Serves arrive every twenty seconds or so.  A stretch of play-camera time far longer than
    that with no accepted start is very unlikely to be real, and it is exactly what the head
    produces on a broadcast whose scores are globally depressed.  Inside such a hole the best
    sub-threshold peak is accepted, but only if a racket impact is audible next to it, which
    keeps the relaxation from firing on dead time.
    """
    if gap_seconds <= 0 or not len(times):
        return starts
    accepted = list(starts)
    for _ in range(len(times)):
        accepted.sort()
        bounds = [(times[a], times[b]) for a, b in zip(accepted, accepted[1:])]
        edges = np.where(play_mask)[0]
        if len(edges):
            bounds = (
                [(times[edges[0]] - 0.001, times[accepted[0]])] + bounds if accepted else bounds
            )
            if accepted:
                bounds.append((times[accepted[-1]], times[edges[-1]] + 0.001))
            else:
                bounds = [(times[edges[0]] - 0.001, times[edges[-1]] + 0.001)]
        best = (None, threshold)
        for low, high in bounds:
            if high - low <= gap_seconds:
                continue
            window = np.where(
                (times > low + MINIMUM_START_GAP_SECONDS)
                & (times < high - MINIMUM_START_GAP_SECONDS)
                & play_mask
            )[0]
            if not len(window):
                continue
            candidate = window[int(np.argmax(masked_scores[window]))]
            value = float(masked_scores[candidate])
            if value < best[1]:
                continue
            if len(onsets) and np.min(np.abs(onsets - times[candidate])) > onset_tolerance:
                continue
            best = (int(candidate), value)
        if best[0] is None:
            break
        accepted.append(best[0])
    return sorted(accepted)


def assemble_attempts(
    times: np.ndarray,
    start_scores: np.ndarray,
    end_scores: np.ndarray,
    *,
    start_threshold: float,
    end_threshold: float,
    shots: list[dict],
    score_changes: np.ndarray,
    play_mask: np.ndarray,
    onsets: np.ndarray | None = None,
    last_impact_buffer: float = LAST_IMPACT_BUFFER_SECONDS,
    gap_fill_seconds: float = GAP_FILL_SECONDS,
    gap_fill_threshold: float = GAP_FILL_THRESHOLD,
    observation_scope: str = "predicted",
    play_probabilities: dict[int, float] | None = None,
) -> list[dict]:
    """Assemble candidate attempt spans, not certified physical ending events.

    The end head decides where the attempt ends.  When it never clears its threshold the
    attempt still has to close somewhere, and closing it at the next structural boundary (the
    next start, the end of the shot, the next scoreboard change) is what produced the long
    tail of over-long segments: those boundaries can be a minute away.  The fallback is now
    the last audio onset inside the attempt plus a fixed buffer. That is only a segmentation
    heuristic: an onset need not be a racket impact, and the buffer does not witness a physical
    point ending. Structural ceilings always bound the emitted span; an insufficient tail
    cannot be lengthened across its camera cut to meet the minimum duration.

    Opt-in structural_context keeps the predicted ending as uncertain metadata
    and extracts to a structural horizon instead. Short intervals remain rows
    with insufficient_context status; no minimum-duration padding is applied.
    """
    if observation_scope not in {"predicted", "structural_context"}:
        raise ValueError("unknown attempt observation scope")
    onsets = np.empty(0) if onsets is None else np.asarray(onsets)
    masked = np.where(play_mask, start_scores, 0.0)
    starts = peaks(masked, times, threshold=start_threshold, minimum_gap=MINIMUM_START_GAP_SECONDS)
    starts = fill_start_gaps(
        starts,
        masked,
        times,
        play_mask,
        onsets=onsets,
        gap_seconds=gap_fill_seconds,
        threshold=gap_fill_threshold,
    )
    shot_ends = np.array([float(shot["t_end"]) for shot in shots])
    attempts = []
    for position, index in enumerate(starts):
        begin = float(times[index])
        following = float(times[starts[position + 1]]) if position + 1 < len(starts) else np.inf
        shot_end = (
            float(shot_ends[np.searchsorted(shot_ends, begin, side="right")])
            if (np.searchsorted(shot_ends, begin, side="right") < len(shot_ends))
            else np.inf
        )
        after_score = score_changes[score_changes > begin + MINIMUM_ATTEMPT_SECONDS]
        score_end = float(after_score[0]) if len(after_score) else np.inf
        ceiling = min(following, shot_end, score_end, begin + MAXIMUM_ATTEMPT_SECONDS)
        insufficient = ceiling < begin + MINIMUM_ATTEMPT_SECONDS
        if insufficient and observation_scope == "predicted":
            continue
        window = np.where((times > begin + MINIMUM_ATTEMPT_SECONDS) & (times <= ceiling))[0]
        best = float(end_scores[window].max()) if len(window) else 0.0
        if len(window) and best >= end_threshold:
            finish = float(times[window[int(np.argmax(end_scores[window]))]])
        else:
            inside = onsets[
                (onsets >= begin) & (onsets <= min(ceiling, begin + MAXIMUM_ATTEMPT_SECONDS))
            ]
            if len(inside):
                finish = float(inside[-1]) + last_impact_buffer
            else:
                finish = begin + MINIMUM_ATTEMPT_SECONDS + last_impact_buffer
            finish = min(finish, ceiling if np.isfinite(ceiling) else finish)
        if not np.isfinite(finish):
            finish = begin + MINIMUM_ATTEMPT_SECONDS
        scope_fields = {}
        predicted_finish = None if insufficient else max(finish, begin + MINIMUM_ATTEMPT_SECONDS)
        if observation_scope == "structural_context":
            from cv.pipeline.attempt_observation_scope import structural_horizon

            context = structural_horizon(
                begin,
                following=following,
                score_end=score_end,
                maximum_duration=MAXIMUM_ATTEMPT_SECONDS,
                shots=shots,
                play_probabilities=play_probabilities or {},
                onsets=onsets,
            )
            finish = context["end"]
            scope_fields = dict(
                observation_scope=observation_scope,
                predicted_rally_t_end=round(predicted_finish, 3)
                if predicted_finish is not None
                else None,
                observation_horizon_reason=context["reason"],
                observation_camera_transition=context["camera_transition"],
                observation_audio_anchor=context["audio_anchor"],
                observation_context_status=(
                    "insufficient_context"
                    if finish - begin < MINIMUM_ATTEMPT_SECONDS
                    else "context_only_physical_ending_unresolved"
                ),
            )
        else:
            finish = predicted_finish
        attempts.append(
            {
                "rally_t_start": round(begin, 3),
                "rally_t_end": round(finish, 3),
                "start_score": round(float(start_scores[index]), 4),
                "end_score": round(best, 4),
                **scope_fields,
            }
        )
    return attempts


# ---------------- per-span assembly ---------------------------------------------------------


# Appending per-broadcast z-scores of every column was measured on the twelve development
# windows and rejected: leave-one-broadcast-out one-to-one matches fell from 261/27/55 to
# 247/24/69 and boundary agreement at 0.5 s from 111 to 84 true positives.  The raw features
# are kept.


def score_change_times(path: Path | None) -> np.ndarray:
    """Times at which a collapsed score-run file reports a different scoreboard state."""
    if path is None or not Path(path).is_file():
        return np.empty(0)
    rows = read_csv_rows(Path(path))
    fields = ("g1", "g2", "p1", "p2", "sets1", "sets2")
    if rows and "server" in rows[0]:
        fields = ("server", "games_1", "games_2", "points_1", "points_2", "completed_sets")
    key = "t_start" if rows and "t_start" in rows[0] else "t"
    times, previous = [], None
    for row in rows:
        state = tuple(row.get(field, "") for field in fields)
        if previous is not None and state != previous:
            times.append(float(row[key]))
        previous = state
    return np.asarray(times)


def _load_audio(out_dir: Path) -> dict:
    """Span-local flux if present, else the runner's whole-match ``serve_audio`` cache."""
    local = out_dir / "audio_flux_v1.npz"
    if local.is_file():
        stored = np.load(local)
        return {
            "z": stored["z"],
            "hop_seconds": float(stored["hop_seconds"]),
            "start_seconds": float(stored["start_seconds"]),
        }
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from contacts_audio import HOP_SAMPLES, SAMPLE_RATE  # noqa: PLC0415

    from cv.pipeline.serve_audio import robust_z  # noqa: PLC0415

    stored = np.load(out_dir / "serve_audio_flux_v1.npz")
    return {
        "z": robust_z(stored["flux"]).astype(np.float32),
        "hop_seconds": HOP_SAMPLES / SAMPLE_RATE,
        "start_seconds": 0.0,
    }


def load_toss(out_dir: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Cached per-candidate toss descriptors, or (None, None) when the tracker never ran."""
    path = out_dir / "toss_windows_v1.npz"
    if not path.is_file():
        return None, None
    stored = np.load(path)
    return stored["times"], stored["values"]


def toss_descriptors(
    track: list[tuple[float, float, float]],
    candidate_time: float,
    *,
    frame_height: float = 1080.0,
    frame_width: float = 1920.0,
) -> list[float]:
    """Describe the ball's vertical arc in one candidate window.

    ``track`` is (time, x, y) in native pixels with y down.  A toss shows as a run where y
    falls (the ball rises), reaches an apex, and then climbs again into the strike; the strike
    itself is the largest frame-to-frame displacement just after the apex.
    """
    if len(track) < 4:
        return list(NO_TOSS)
    track = sorted(track)
    seconds = np.array([row[0] for row in track])
    xs = np.array([row[1] for row in track])
    ys = np.array([row[2] for row in track])
    span = seconds[-1] - seconds[0]
    detection_rate = float(len(track) / max(1.0, span * 50.0))
    apex = int(np.argmin(ys))
    rise = float((ys[:apex].max() - ys[apex]) / frame_height) if apex > 0 else 0.0
    fall = float((ys[apex:].max() - ys[apex]) / frame_height) if apex < len(ys) - 1 else 0.0
    steps = np.hypot(np.diff(xs), np.diff(ys))
    after = steps[apex:] if apex < len(steps) else steps[-1:]
    return [
        1.0,
        float(seconds[apex] - candidate_time),
        min(detection_rate, 2.0),
        rise,
        float(seconds[apex] - seconds[0]),
        min(rise, fall),
        float(after.max() / frame_width) if len(after) else 0.0,
        float(1.0 - ys[apex] / frame_height),
    ]


def play_time_references(
    signals: dict, audio: dict, shots: list[dict], view_probabilities: dict
) -> tuple[float, float]:
    """Per-broadcast audio and motion scales measured on play-camera time only."""
    fps = float(signals["fps"])
    offset = float(signals["start_seconds"])
    motion = signals["motion"]
    audio_z = audio["z"]
    audio_fps = 1.0 / audio["hop_seconds"]
    audio_offset = audio["start_seconds"]
    motion_parts, audio_parts = [], []
    for index, shot in enumerate(shots):
        if view_probabilities.get(index, {}).get("play", 0.0) < 0.35:
            continue
        low, high = float(shot["t_start"]), float(shot["t_end"])
        motion_parts.append(_uniform_window(motion, fps, offset, low, high))
        audio_parts.append(_uniform_window(audio_z, audio_fps, audio_offset, low, high))
    if not motion_parts:
        motion_parts = [motion]
        audio_parts = [audio_z]
    motion_reference = float(np.median(np.concatenate(motion_parts)))
    audio_reference = float(np.percentile(np.concatenate(audio_parts), 90))
    return max(1e-3, audio_reference), max(1e-3, motion_reference)


def assemble_span_features(
    out_dir: Path, *, score_runs: Path | None = None, cache: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Feature matrix, grid times and per-step play-view mask for one extracted span."""
    import json

    from cv.pipeline import view_classifier

    cache_path = out_dir / "serve_features_v1.npz"
    audio_input = out_dir / "audio_flux_v1.npz"
    if not audio_input.is_file():
        audio_input = out_dir / "serve_audio_flux_v1.npz"
    feature_inputs = [
        out_dir / name
        for name in (
            "shot_signals_v1.npz",
            "shot_boundaries_v1.csv",
            "shot_views_v1.csv",
            "player_boxes_v1.npz",
            "frames_native.json",
        )
    ]
    feature_inputs.append(audio_input)
    if (out_dir / "toss_windows_v1.npz").is_file():
        feature_inputs.append(out_dir / "toss_windows_v1.npz")
    if score_runs is not None:
        if not score_runs.is_file():
            raise FileNotFoundError(score_runs)
        feature_inputs.append(score_runs)
    receipt = {
        "out_dir": out_dir,
        "stage": "serve_features",
        "command": ["python", str(Path(__file__).resolve())],
        "configuration": {
            "entrypoint": "cv.pipeline.serve_detector.assemble_span_features",
            "score_runs": str(score_runs) if score_runs else None,
        },
        "inputs": feature_inputs,
        "outputs": [cache_path],
    }
    if cache and stage_receipt_matches(**receipt):
        with np.load(cache_path) as stored:
            return stored["features"], stored["times"], stored["play_mask"]
    signals = dict(np.load(out_dir / "shot_signals_v1.npz"))
    audio = _load_audio(out_dir)
    shots = view_classifier.read_shots(out_dir / "shot_boundaries_v1.csv")
    views = read_csv_rows(out_dir / "shot_views_v1.csv")
    view_probabilities = {
        int(row["shot_index"]): {name: float(row[name]) for name in view_classifier.VIEW_CLASSES}
        for row in views
    }
    player_rows = np.load(out_dir / "player_boxes_v1.npz")["rows"]
    meta = json.loads((out_dir / "frames_native.json").read_text())
    player_times = view_classifier.frame_times(
        len(player_rows), fps=meta["fps"], start_seconds=meta["start_seconds"]
    )
    start = float(signals["start_seconds"])
    end = start + len(signals["motion"]) / float(signals["fps"])
    # Feature windows already use available source support at either boundary.
    # Do not delete starts in the first/last three seconds of an extracted span.
    # The last motion sample, not the exclusive span end, bounds candidate time.
    # Retain the old interior arithmetic origin: rebasing by three seconds
    # changes floating-point rounding at integer feature-window boundaries.
    origin = start + 3.0
    first_step = -round(3.0 / GRID_SECONDS)
    last_sample = end - 1.0 / float(signals["fps"])
    last_step = int(np.floor((last_sample - origin) / GRID_SECONDS))
    times = origin + np.arange(first_step, last_step + 1) * GRID_SECONDS
    if len(times):
        times[0] = start
    shot_starts = np.array([float(shot["t_start"]) for shot in shots])
    play_mask = np.zeros(len(times), dtype=bool)
    for index, moment in enumerate(times):
        shot_index = min(
            max(int(np.searchsorted(shot_starts, moment, side="right") - 1), 0), len(shots) - 1
        )
        play_mask[index] = view_probabilities.get(shot_index, {}).get("play", 0.0) >= 0.35
    audio_reference, motion_reference = play_time_references(
        signals, audio, shots, view_probabilities
    )
    toss = load_toss(out_dir)
    features = build_features(
        times,
        signals=signals,
        audio_z=audio["z"],
        audio_hop=audio["hop_seconds"],
        audio_offset=audio["start_seconds"],
        shots=shots,
        view_probabilities=view_probabilities,
        player_rows=player_rows,
        player_times=player_times,
        score_changes=score_change_times(score_runs),
        audio_reference=audio_reference,
        motion_reference=motion_reference,
        toss_times=toss[0],
        toss_values=toss[1],
    )
    if cache:
        np.savez_compressed(cache_path, features=features, times=times, play_mask=play_mask)
        write_stage_receipt(**receipt)
    return features, times, play_mask


def predict_span(
    out_dir: Path,
    features: np.ndarray,
    times: np.ndarray,
    play_mask: np.ndarray,
    start_model,
    end_model,
    *,
    start_threshold: float,
    end_threshold: float,
    score_runs: Path | None = None,
    observation_scope: str = "predicted",
) -> list[dict]:
    from cv.pipeline import view_classifier

    audio = _load_audio(out_dir)
    onsets = audio["start_seconds"] + np.where(audio["z"] >= 5.0)[0] * audio["hop_seconds"]
    return assemble_attempts(
        times,
        start_model.predict_proba(features)[:, 1],
        end_model.predict_proba(features)[:, 1],
        start_threshold=start_threshold,
        end_threshold=end_threshold,
        shots=view_classifier.read_shots(out_dir / "shot_boundaries_v1.csv"),
        score_changes=score_change_times(score_runs),
        play_mask=play_mask,
        onsets=onsets,
        observation_scope=observation_scope,
        play_probabilities={
            int(row["shot_index"]): float(row["play"])
            for row in read_csv_rows(out_dir / "shot_views_v1.csv")
        }
        if observation_scope == "structural_context"
        else None,
    )


def save_head(path: Path, start_model, end_model, metadata: dict) -> None:
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(
            {
                "schema": "serve_detector_heads_v1",
                "features": feature_names(),
                "start": start_model,
                "end": end_model,
                "metadata": metadata,
            },
            handle,
        )


def load_head(path: Path) -> dict:
    import pickle

    with path.open("rb") as handle:
        return pickle.load(handle)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_attempts(path: Path, attempts: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from cv.pipeline.attempt_observation_scope import SCOPE_FIELDS

    fields = ["rally_t_start", "rally_t_end", "start_score", "end_score"]
    fields.extend(key for key in SCOPE_FIELDS if any(key in row for row in attempts))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(attempts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("boxes", "predict"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--score-runs", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--observation-scope",
        choices=("predicted", "structural_context"),
        default="predicted",
        help="optional source context horizon; preserves tentative end prediction separately",
    )
    args = parser.parse_args()
    if args.action == "boxes":
        if args.weights is None:
            parser.error("--weights is required for boxes")
        frames = sorted((args.out / "frames_native").glob("n_*.jpg"))
        rows = player_boxes(frames, weights=args.weights, device=args.device)
        np.savez_compressed(args.out / "player_boxes_v1.npz", rows=rows)
        print(f"{rows.shape} -> {args.out / 'player_boxes_v1.npz'}")
        return 0
    if args.model is None:
        parser.error("--model is required for predict")
    bundle = load_head(args.model)
    features, times, play_mask = assemble_span_features(args.out, score_runs=args.score_runs)
    attempts = predict_span(
        args.out,
        features,
        times,
        play_mask,
        bundle["start"],
        bundle["end"],
        start_threshold=bundle["metadata"]["start_threshold"],
        end_threshold=bundle["metadata"]["end_threshold"],
        score_runs=args.score_runs,
        observation_scope=args.observation_scope,
    )
    path = args.out / "serve_attempts_v1.csv"
    write_attempts(path, attempts)
    print(f"{len(attempts)} attempts -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
