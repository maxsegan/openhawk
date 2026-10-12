from pathlib import Path

import numpy as np
import pytest

from cv.pipeline import serve_speed_graphic as speed


def reading(frame: int, value: int | None, confidence: float = 0.9) -> speed.FrameReading:
    return speed.FrameReading(
        frame,
        value,
        confidence if value is not None else 0.0,
        value is None,
        None if value is not None else "no_read",
        {},
        (0, 0, 10, 10),
    )


def test_association_rejects_stale_previous_serve() -> None:
    rows = [reading(frame, 106) for frame in range(60, 65)]
    rows += [reading(frame, 106) for frame in range(65, 70)]
    rows += [reading(frame, 121) for frame in range(70, 75)]

    result = speed.associate_serve(rows, 64.5, speed.PROFILES["uso2020_ibm"])

    assert result["display_value"] == 121
    assert result["first_display_frame"] == 70
    assert result["previous_stable_value"] == 106
    assert result["speed_mps"] == pytest.approx(121 * speed.MPH_TO_MPS)


def test_association_marks_persistent_same_value_as_weaker() -> None:
    rows = [reading(frame, 175) for frame in range(1, 10)]
    result = speed.associate_serve(rows, 5.0, speed.PROFILES["ao2023_tmgm"])
    assert not result["abstained"]
    assert result["association"] == "persistent_same_value_without_observed_refresh"
    assert result["confidence"] <= 0.6


def test_association_allows_delayed_update() -> None:
    rows = [reading(frame, 175) for frame in range(1, 9)]
    rows += [reading(frame, 132) for frame in range(9, 14)]
    result = speed.associate_serve(rows, 5.0, speed.PROFILES["ao2023_tmgm"])
    assert result["display_value"] == 132
    assert result["visible_refresh_observed"]


def test_frame_reader_uses_rendering_consensus(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = iter([("175", 0.8), ("175", 0.7), ("176", 0.9), ("175", 0.6)])
    monkeypatch.setattr(speed.shutil, "which", lambda _name: "/usr/bin/tesseract")
    monkeypatch.setattr(speed, "_ocr_digits", lambda _image, _executable: next(outputs))
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    result = speed.read_frame(image, 10, speed.PROFILES["ao2023_tmgm"])

    assert result.value == 175
    assert not result.abstained
    assert 0 < result.confidence <= 1


def test_frame_reader_abstains_when_ocr_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(speed.shutil, "which", lambda _name: None)
    result = speed.read_frame(
        np.zeros((1080, 1920, 3), dtype=np.uint8), 1, speed.PROFILES["ao2023_tmgm"]
    )
    assert result.abstention_reason == "ocr_executable_unavailable"


def test_frame_name_is_strict() -> None:
    assert speed._frame_number(Path("f_0042.jpg")) == 42
    with pytest.raises(ValueError):
        speed._frame_number(Path("42.jpg"))


def test_score_overlay_parser_keeps_two_rows_and_crop_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    words = [
        speed.OCRWord(">", 0.94, 2, 20, 10, 18),
        speed.OCRWord("RYBAKINA", 0.96, 20, 20, 90, 18),
        speed.OCRWord("6", 0.97, 180, 20, 12, 18),
        speed.OCRWord("0", 0.98, 220, 20, 12, 18),
        speed.OCRWord("15", 0.99, 270, 20, 24, 18),
        speed.OCRWord("SABALENKA", 0.95, 20, 54, 100, 18),
        speed.OCRWord("4", 0.97, 180, 54, 12, 18),
        speed.OCRWord("1", 0.98, 220, 54, 12, 18),
        speed.OCRWord("30", 0.99, 270, 54, 24, 18),
    ]
    monkeypatch.setattr(speed.shutil, "which", lambda _name: "/usr/bin/tesseract")
    monkeypatch.setattr(speed, "_ocr_words", lambda _image, _executable: words)

    result = speed.read_score_overlay(np.zeros((1080, 1920, 3), dtype=np.uint8), 68)

    assert not result.abstained
    assert result.crop_sha256
    assert result.rows[0] == {
        "name": "RYBAKINA",
        "values": ["6", "0"],
        "points": "15",
    }
    assert result.rows[1]["points"] == "30"
    assert result.serving_row == 1


def test_score_overlay_abstains_without_two_score_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(speed.shutil, "which", lambda _name: "/usr/bin/tesseract")
    monkeypatch.setattr(
        speed,
        "_ocr_words",
        lambda _image, _executable: [speed.OCRWord("MELBOURNE", 0.9, 1, 1, 20, 10)],
    )
    result = speed.read_score_overlay(np.zeros((1080, 1920, 3), dtype=np.uint8), 1)
    assert result.abstained
    assert result.abstention_reason == "no_two_row_score_read"
