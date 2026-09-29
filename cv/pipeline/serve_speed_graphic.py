"""Read a courtside/broadcast serve-speed graphic from native video frames.

The reader is deliberately small and fail-closed.  A broadcast-family profile
defines only the stable native-frame rectangle and displayed unit; digit OCR is
run on several deterministic grayscale/threshold renderings and a value is
emitted only when the renderings and consecutive native frames agree.  When a
previous speed is already on the board, the serve is associated with the first
stable *changed* value after contact. Courtside boards can update well into the
rally, so association is deliberately not cut off at the return. If no visible
refresh occurs in the supplied audit window, the reader returns the persistent
value as explicitly weaker evidence.

Profiles are source-layout configuration, not evaluation labels.  The CLI binds
the source video and every inspected native frame, reports confidence and an
abstention reason, and never changes frame timestamps.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

from cv.pipeline import paths, provenance

MPH_TO_MPS = 0.44704
KMH_TO_MPS = 1.0 / 3.6


@dataclass(frozen=True)
class GraphicProfile:
    """One stable speed-board layout in normalized native coordinates."""

    name: str
    roi_xyxy: tuple[float, float, float, float]
    unit: str
    minimum: int
    maximum: int
    thresholds: tuple[int, ...]

    def validate(self) -> None:
        x0, y0, x1, y1 = self.roi_xyxy
        if (
            self.unit not in {"km/h", "mph"}
            or not 0 <= x0 < x1 <= 1
            or not 0 <= y0 < y1 <= 1
            or not 0 < self.minimum < self.maximum
            or not self.thresholds
            or any(not 0 < value < 255 for value in self.thresholds)
        ):
            raise ValueError("invalid serve-speed graphic profile")


PROFILES = {
    "ao2023_tmgm": GraphicProfile(
        "ao2023_tmgm",
        (1585 / 1920, 120 / 1080, 1675 / 1920, 162 / 1080),
        "km/h",
        100,
        240,
        (100, 140, 180),
    ),
    "rg2024_infosys": GraphicProfile(
        "rg2024_infosys",
        (380 / 1920, 203 / 1080, 455 / 1920, 238 / 1080),
        "km/h",
        100,
        240,
        (140, 160, 180),
    ),
    "uso2020_ibm": GraphicProfile(
        "uso2020_ibm",
        (555 / 1920, 90 / 1080, 625 / 1920, 128 / 1080),
        "mph",
        60,
        160,
        (70, 85, 100),
    ),
}


@dataclass(frozen=True)
class FrameReading:
    frame: int
    value: int | None
    confidence: float
    abstained: bool
    abstention_reason: str | None
    votes: dict[str, int]
    roi_native_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class OCRWord:
    """One Tesseract token in coordinates relative to the inspected crop."""

    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class ScoreOverlayReading:
    """Fail-closed two-row tennis-score read with an auditable native crop."""

    frame: int
    present: bool
    rows: tuple[dict[str, Any], ...]
    serving_row: int | None
    confidence: float
    abstained: bool
    abstention_reason: str | None
    roi_native_xyxy: tuple[int, int, int, int] | None
    crop_sha256: str | None
    raw_text: str
    words: tuple[OCRWord, ...]


# Broadcast scoreboards are overwhelmingly corner graphics.  These are search
# regions, not match-specific labels: Tesseract must still find two plausible
# tennis-score rows and all failed regions remain in the receipt.
SCORE_SEARCH_ROIS = (
    (0.00, 0.76, 0.42, 1.00),
    (0.58, 0.76, 1.00, 1.00),
    (0.00, 0.62, 0.50, 1.00),
    (0.50, 0.62, 1.00, 1.00),
    (0.00, 0.00, 0.50, 0.38),
    (0.50, 0.00, 1.00, 0.38),
)
POINT_TOKENS = {"0", "15", "30", "40", "AD"}
SERVE_MARKER_TOKENS = {">", "►", "▶", "•", "●", "*"}


def _native_roi(image: np.ndarray, profile: GraphicProfile) -> tuple[np.ndarray, tuple[int, ...]]:
    profile.validate()
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 100:
        raise ValueError("one native BGR frame is required")
    height, width = image.shape[:2]
    x0, y0, x1, y1 = profile.roi_xyxy
    box = (
        int(round(x0 * width)),
        int(round(y0 * height)),
        int(round(x1 * width)),
        int(round(y1 * height)),
    )
    left, top, right, bottom = box
    crop = image[top:bottom, left:right]
    if not crop.size:
        raise ValueError("serve-speed ROI is empty")
    return crop, box


def _ocr_digits(image: np.ndarray, executable: str) -> tuple[str, float]:
    enlarged = cv2.resize(image, None, fx=8, fy=8, interpolation=cv2.INTER_LANCZOS4)
    ok, encoded = cv2.imencode(".png", enlarged)
    if not ok:
        return "", 0.0
    result = subprocess.run(
        [
            executable,
            "stdin",
            "stdout",
            "--psm",
            "7",
            "-c",
            "tessedit_char_whitelist=0123456789",
            "tsv",
        ],
        input=encoded.tobytes(),
        capture_output=True,
        check=False,
        timeout=10,
    )
    if result.returncode:
        return "", 0.0
    rows = result.stdout.decode("utf-8", errors="replace").splitlines()[1:]
    words: list[tuple[str, float]] = []
    for row in rows:
        fields = row.split("\t")
        if len(fields) != 12:
            continue
        text = "".join(character for character in fields[11] if character.isdigit())
        if not text:
            continue
        try:
            confidence = float(fields[10])
        except ValueError:
            continue
        words.append((text, confidence))
    if not words:
        return "", 0.0
    text, confidence = max(words, key=lambda row: (len(row[0]), row[1]))
    return text, float(np.clip(confidence / 100.0, 0.0, 1.0))


def _ocr_words(image: np.ndarray, executable: str) -> list[OCRWord]:
    """Return Tesseract TSV words for a scoreboard search crop."""
    enlarged = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    ok, encoded = cv2.imencode(".png", enlarged)
    if not ok:
        return []
    result = subprocess.run(
        [executable, "stdin", "stdout", "--psm", "11", "tsv"],
        input=encoded.tobytes(),
        capture_output=True,
        check=False,
        timeout=15,
    )
    if result.returncode:
        return []
    output: list[OCRWord] = []
    for row in result.stdout.decode("utf-8", errors="replace").splitlines()[1:]:
        fields = row.split("\t")
        if len(fields) != 12 or not fields[11].strip():
            continue
        try:
            confidence = float(fields[10]) / 100.0
            left, top, width, height = (round(int(fields[index]) / 2) for index in range(6, 10))
        except ValueError:
            continue
        if confidence < 0:
            continue
        output.append(
            OCRWord(
                text=fields[11].strip(),
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                left=left,
                top=top,
                width=width,
                height=height,
            )
        )
    return output


def _score_token(text: str) -> str | None:
    token = re.sub(r"[^0-9A-Za-z]", "", text).upper()
    if token in POINT_TOKENS:
        return token
    return token if token.isdigit() and len(token) <= 2 else None


def _group_score_rows(
    words: list[OCRWord],
) -> tuple[tuple[dict[str, Any], ...], float, int | None] | None:
    """Parse two nearby OCR lines into the raw score-reader row contract."""
    useful = [word for word in words if word.text.strip()]
    if len(useful) < 4:
        return None
    useful.sort(key=lambda word: (word.top + word.height / 2, word.left))
    groups: list[list[OCRWord]] = []
    for word in useful:
        center = word.top + word.height / 2
        target = next(
            (
                group
                for group in groups
                if abs(center - np.median([item.top + item.height / 2 for item in group]))
                <= max(8.0, 0.75 * np.median([item.height for item in group]))
            ),
            None,
        )
        (target if target is not None else groups.append([]) or groups[-1]).append(word)

    candidates: list[tuple[list[OCRWord], list[str]]] = []
    for group in groups:
        ordered = sorted(group, key=lambda word: word.left)
        tokens = [token for word in ordered if (token := _score_token(word.text)) is not None]
        if tokens and any(token in POINT_TOKENS for token in tokens):
            candidates.append((ordered, tokens))
    if len(candidates) < 2:
        return None
    # A scoreboard's two rows are close, similarly tall, and carry similar token counts.
    best: tuple[float, tuple[list[OCRWord], list[str]], tuple[list[OCRWord], list[str]]] | None = (
        None
    )
    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
            y1 = np.median([word.top + word.height / 2 for word in first[0]])
            y2 = np.median([word.top + word.height / 2 for word in second[0]])
            separation = abs(float(y2 - y1))
            height = float(np.median([word.height for word in first[0] + second[0]]))
            if separation < 0.5 * height or separation > 4.5 * height:
                continue
            score = (
                min(len(first[1]), len(second[1]))
                - 0.4 * abs(len(first[1]) - len(second[1]))
                + float(np.mean([word.confidence for word in first[0] + second[0]]))
            )
            if best is None or score > best[0]:
                best = (score, first, second)
    if best is None:
        return None

    selected = sorted(best[1:], key=lambda item: np.median([word.top for word in item[0]]))
    rows = []
    confidences = []
    marker_rows = []
    for row_index, (row_words, tokens) in enumerate(selected, start=1):
        point = tokens[-1]
        if point not in POINT_TOKENS or len(tokens) < 2:
            return None
        values = tokens[:-1]
        names = [
            re.sub(r"[^A-Za-z'-]", "", word.text).upper()
            for word in row_words
            if any(character.isalpha() for character in word.text)
            and _score_token(word.text) is None
        ]
        rows.append({"name": " ".join(filter(None, names)), "values": values, "points": point})
        confidences.extend(word.confidence for word in row_words)
        if any(word.text.strip() in SERVE_MARKER_TOKENS for word in row_words):
            marker_rows.append(row_index)
    confidence = float(np.clip(np.median(confidences) * min(1.0, best[0] / 4.0), 0.0, 1.0))
    serving_row = marker_rows[0] if len(marker_rows) == 1 else None
    return (tuple(rows), confidence, serving_row) if confidence >= 0.30 else None


def read_score_overlay(
    image: np.ndarray,
    frame: int,
    *,
    tesseract: str = "tesseract",
    search_rois: Iterable[tuple[float, float, float, float]] = SCORE_SEARCH_ROIS,
) -> ScoreOverlayReading:
    """Locate and OCR a two-row tennis scoreboard in one native frame.

    A serve marker is emitted only when Tesseract returns exactly one unambiguous
    marker glyph on one of the parsed rows. Ball icons that OCR does not preserve,
    and ambiguous or absent markers, leave ``serving_row`` null.
    """
    if image.ndim != 3 or image.shape[2] != 3 or frame < 1:
        raise ValueError("positive frame and one native BGR image required")
    executable = shutil.which(tesseract)
    if executable is None:
        return ScoreOverlayReading(
            frame, False, (), None, 0.0, True, "ocr_executable_unavailable", None, None, "", ()
        )
    height, width = image.shape[:2]
    attempts = []
    inspected = []
    for normalized in search_rois:
        x0, y0, x1, y1 = normalized
        if not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1:
            raise ValueError("score search ROI must be normalized and ordered")
        box = (
            int(round(x0 * width)),
            int(round(y0 * height)),
            int(round(x1 * width)),
            int(round(y1 * height)),
        )
        left, top, right, bottom = box
        crop = image[top:bottom, left:right]
        words = _ocr_words(crop, executable)
        inspected.append((box, crop, words))
        parsed = _group_score_rows(words)
        if parsed is not None:
            rows, confidence, serving_row = parsed
            attempts.append((confidence, box, crop, rows, serving_row, words))
    if not attempts:
        box, crop, words = max(
            inspected,
            key=lambda item: (
                sum(_score_token(word.text) is not None for word in item[2]),
                sum(word.confidence for word in item[2]),
            ),
        )
        return ScoreOverlayReading(
            frame,
            False,
            (),
            None,
            0.0,
            True,
            "no_two_row_score_read",
            box,
            hashlib.sha256(np.ascontiguousarray(crop).tobytes()).hexdigest(),
            " ".join(word.text for word in words),
            tuple(words),
        )
    confidence, box, crop, rows, serving_row, words = max(attempts, key=lambda row: row[0])
    crop_hash = hashlib.sha256(np.ascontiguousarray(crop).tobytes()).hexdigest()
    raw_text = "\n".join(" ".join(word.text for word in group) for group in [list(words)])
    return ScoreOverlayReading(
        frame=frame,
        present=True,
        rows=rows,
        serving_row=serving_row,
        confidence=confidence,
        abstained=False,
        abstention_reason=None,
        roi_native_xyxy=box,
        crop_sha256=crop_hash,
        raw_text=raw_text,
        words=tuple(words),
    )


def read_frame(
    image: np.ndarray,
    frame: int,
    profile: GraphicProfile,
    *,
    tesseract: str = "tesseract",
) -> FrameReading:
    """OCR one native frame; agreement across renderings is mandatory."""
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 1:
        raise ValueError("positive one-based native frame required")
    executable = shutil.which(tesseract)
    crop, box = _native_roi(image, profile)
    if executable is None:
        return FrameReading(frame, None, 0.0, True, "ocr_executable_unavailable", {}, box)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    renderings: dict[str, np.ndarray] = {"gray": gray}
    for threshold in profile.thresholds:
        renderings[f"threshold_{threshold}"] = cv2.threshold(
            gray, threshold, 255, cv2.THRESH_BINARY
        )[1]
    votes: dict[str, int] = {}
    confidences: dict[int, list[float]] = {}
    for name, rendered in renderings.items():
        text, confidence = _ocr_digits(rendered, executable)
        if len(text) != 3:
            continue
        value = int(text)
        if not profile.minimum <= value <= profile.maximum:
            continue
        votes[name] = value
        confidences.setdefault(value, []).append(confidence)
    if not votes:
        return FrameReading(frame, None, 0.0, True, "no_valid_three_digit_read", {}, box)
    counts = Counter(votes.values())
    value, support = counts.most_common(1)[0]
    tied = sum(count == support for count in counts.values()) > 1
    if tied or support < 2:
        return FrameReading(frame, None, 0.0, True, "ocr_renderings_disagree", votes, box)
    agreement = support / len(renderings)
    engine_confidence = float(np.median(confidences[value]))
    # Tesseract sometimes assigns zero confidence to small perspective digits
    # despite unanimous preprocessing agreement. Agreement remains observable,
    # but cannot by itself produce a high-confidence frame.
    confidence = float(np.clip(0.65 * agreement + 0.35 * engine_confidence, 0.0, 1.0))
    return FrameReading(frame, value, confidence, False, None, votes, box)


def _runs(readings: Iterable[FrameReading], maximum_gap: int) -> list[list[FrameReading]]:
    runs: list[list[FrameReading]] = []
    for row in readings:
        if row.abstained or row.value is None:
            continue
        if (
            runs
            and runs[-1][-1].value == row.value
            and row.frame - runs[-1][-1].frame <= maximum_gap
        ):
            runs[-1].append(row)
        else:
            runs.append([row])
    return runs


def associate_serve(
    readings: Iterable[FrameReading],
    contact_frame: float,
    profile: GraphicProfile,
    *,
    minimum_stable_frames: int = 3,
    maximum_gap: int = 2,
) -> dict[str, Any]:
    """Associate the first stable board update after contact with one serve."""
    rows = sorted(readings, key=lambda row: row.frame)
    if (
        not math.isfinite(contact_frame)
        or contact_frame < 1
        or minimum_stable_frames < 2
        or maximum_gap < 1
    ):
        raise ValueError("finite contact and positive association settings required")
    pre_runs = [run for run in _runs((r for r in rows if r.frame < contact_frame), maximum_gap)]
    stable_pre = [run for run in pre_runs if len(run) >= minimum_stable_frames]
    previous = stable_pre[-1][-1].value if stable_pre else None
    post_runs = [run for run in _runs((r for r in rows if r.frame >= contact_frame), maximum_gap)]
    stable_post = [run for run in post_runs if len(run) >= minimum_stable_frames]
    changed = [run for run in stable_post if previous is None or run[0].value != previous]
    persistent = [run for run in stable_post if previous is not None and run[0].value == previous]
    if not changed and not persistent:
        reason = (
            "no_stable_post_contact_read" if previous is None else "no_persistent_post_contact_read"
        )
        return {
            "schema": "serve_speed_graphic_reading_v1",
            "profile": profile.name,
            "unit": profile.unit,
            "display_value": None,
            "speed_mps": None,
            "contact_frame": float(contact_frame),
            "first_display_frame": None,
            "frames_after_contact": None,
            "confidence": 0.0,
            "abstained": True,
            "abstention_reason": reason,
            "previous_stable_value": previous,
            "association": "first stable changed value after contact",
        }
    selected = changed[0] if changed else max(persistent, key=len)
    association = (
        "first_stable_changed_value_in_audit_window"
        if changed and previous is not None
        else "first_stable_post_contact_value_no_readable_predecessor"
        if changed
        else "persistent_same_value_without_observed_refresh"
    )
    value = int(selected[0].value)
    conversion = MPH_TO_MPS if profile.unit == "mph" else KMH_TO_MPS
    temporal_support = min(len(selected) / (minimum_stable_frames + 2), 1.0)
    confidence = float(
        np.clip(np.median([row.confidence for row in selected]) * temporal_support, 0.0, 1.0)
    )
    if not changed:
        confidence = min(confidence, 0.60)
    return {
        "schema": "serve_speed_graphic_reading_v1",
        "profile": profile.name,
        "unit": profile.unit,
        "display_value": value,
        "speed_mps": value * conversion,
        "contact_frame": float(contact_frame),
        "first_display_frame": selected[0].frame,
        "frames_after_contact": selected[0].frame - float(contact_frame),
        "last_support_frame": selected[-1].frame,
        "stable_support_frames": len(selected),
        "confidence": confidence,
        "abstained": False,
        "abstention_reason": None,
        "previous_stable_value": previous,
        "association": association,
        "visible_refresh_observed": bool(changed and previous is not None),
        "association_interval_end_frame": rows[-1].frame,
    }


def read_paths(
    frame_paths: Mapping[int, Path],
    contact_frame: float,
    profile: GraphicProfile,
) -> tuple[dict[str, Any], list[FrameReading]]:
    readings = []
    for frame, path in sorted(frame_paths.items()):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            readings.append(
                FrameReading(frame, None, 0.0, True, "native_frame_unreadable", {}, (0, 0, 0, 0))
            )
        else:
            readings.append(read_frame(image, frame, profile))
    return associate_serve(readings, contact_frame, profile), readings


def _frame_number(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("f_") or not stem[2:].isdigit():
        raise ValueError(f"native frame name must be f_NNNN: {path}")
    return int(stem[2:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--contact-frame", type=float, required=True)
    parser.add_argument("--first-frame", type=int, required=True)
    parser.add_argument("--last-frame", type=int, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.stride < 1 or args.first_frame > args.last_frame:
        raise ValueError("new output, ordered frame range and positive stride required")
    all_paths = {_frame_number(path): path for path in args.frames_dir.glob("f_*.jpg")}
    selected = {
        frame: all_paths[frame]
        for frame in range(args.first_frame, args.last_frame + 1, args.stride)
        if frame in all_paths
    }
    if not selected:
        raise ValueError("no native frames in requested range")
    reading, frames = read_paths(selected, args.contact_frame, PROFILES[args.profile])
    report = {
        "schema": "serve_speed_graphic_run_v1",
        "automatic": True,
        "status": "abstained" if reading["abstained"] else "measured",
        "reading": reading,
        "profile": asdict(PROFILES[args.profile]),
        "configuration": {
            "first_frame": args.first_frame,
            "last_frame": args.last_frame,
            "stride": args.stride,
            "native_timestamps_changed": False,
            "minimum_stable_frames": 3,
            "association": "changed value, else explicitly weak persistent same value",
        },
        "source_video": provenance.file_record(args.source_video),
        "source_frames": [provenance.file_record(path) for path in selected.values()],
        "implementation": provenance.file_record(Path(__file__)),
        "code": provenance.git_record(paths.REPO_ROOT),
        "human_derived_inputs": [],
        "frame_readings": [asdict(row) for row in frames],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], **reading}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
