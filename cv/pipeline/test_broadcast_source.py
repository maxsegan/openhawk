from pathlib import Path
from types import SimpleNamespace

import pytest

from cv.pipeline import broadcast_source
from cv.pipeline.cadence_normalize import predicted_repeats


def test_probe_uses_container_duration_when_stream_duration_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        broadcast_source.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=(
                '{"streams":[{"width":1920,"height":1080,'
                '"avg_frame_rate":"50/1","r_frame_rate":"50/1"}],'
                '"format":{"duration":"123.5"}}'
            )
        ),
    )

    result = broadcast_source.probe(Path("match.mkv"))

    assert result["duration_seconds"] == 123.5
    assert result["fps"] == 50.0
    assert result["resolution_valid"] is True
    assert result["resolution_status"] == "native_1080_or_better"


def test_probe_prefers_stream_duration(monkeypatch) -> None:
    monkeypatch.setattr(
        broadcast_source.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=(
                '{"streams":[{"width":1920,"height":1080,'
                '"avg_frame_rate":"25/1","r_frame_rate":"25/1",'
                '"duration":"122.0"}],"format":{"duration":"123.5"}}'
            )
        ),
    )

    result = broadcast_source.probe(Path("match.mp4"))

    assert result["duration_seconds"] == 122.0


def test_probe_flags_subnative_source(monkeypatch) -> None:
    monkeypatch.setattr(
        broadcast_source.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=(
                '{"streams":[{"width":1280,"height":720,'
                '"avg_frame_rate":"25/1","r_frame_rate":"25/1",'
                '"duration":"10.0"}],"format":{"duration":"10.0"}}'
            )
        ),
    )

    result = broadcast_source.probe(Path("legacy.mp4"))

    assert result["resolution_valid"] is False
    assert result["resolution_status"] == "LEGACY_BAD_SHOULD_UPDATE"


def test_source_cadence_records_pts_and_heals_regular_conversion(monkeypatch) -> None:
    monkeypatch.setattr(
        broadcast_source, "audit_source_timebase", lambda *a: {"nominal_timeline_valid": True}
    )
    monkeypatch.setattr(broadcast_source, "_sample_starts", lambda *args, **kwargs: [0.0])
    monkeypatch.setattr(
        broadcast_source,
        "_frame_timestamps",
        lambda *args, **kwargs: [index / 60.0 for index in range(300)],
    )
    # 24 unique exposures repeated in a textbook 2:3 conversion to 60 fps.
    kept = []
    frame = 1
    for index in range(120):
        kept.append(frame)
        frame += 2 if index % 2 == 0 else 3
    monkeypatch.setattr(broadcast_source, "_kept_sample_frames", lambda *args, **kwargs: kept)

    result = broadcast_source.source_cadence(
        Path("source.mp4"), {"fps": 60.0, "duration_seconds": 60.0}
    )

    assert result["decision"] == "healed"
    assert result["processing_fps"] == 24.0
    assert result["timestamp_samples"][0]["strictly_increasing"] is True


def test_source_cadence_rejects_nonmonotonic_pts(monkeypatch) -> None:
    monkeypatch.setattr(broadcast_source, "_sample_starts", lambda *args, **kwargs: [0.0])
    monkeypatch.setattr(
        broadcast_source,
        "_frame_timestamps",
        lambda *args, **kwargs: [0.0, 1 / 25.0, 1 / 25.0],
    )

    result = broadcast_source.source_cadence(
        Path("source.mp4"), {"fps": 25.0, "duration_seconds": 60.0}
    )

    assert result["decision"] == "rejected"
    assert result["reason"] == "irregular_source_timestamps"


@pytest.mark.parametrize(
    "kind", ["alternating", "wrong_nominal_fps", "empty_sample", "single_frame"]
)
def test_source_cadence_rejects_unrepresentable_or_missing_pts(monkeypatch, kind):
    timestamps = [index / 50.0 for index in range(300)]
    if kind == "alternating":
        timestamps = [
            value + (0.01 if index % 2 else 0.0) for index, value in enumerate(timestamps)
        ]
    elif kind == "wrong_nominal_fps":
        timestamps = [index / 49.0 for index in range(300)]
    elif kind == "empty_sample":
        timestamps = []
    else:
        timestamps = [0.0]
    monkeypatch.setattr(broadcast_source, "_sample_starts", lambda *a, **kw: [0.0])
    monkeypatch.setattr(broadcast_source, "_frame_timestamps", lambda *a: timestamps)
    monkeypatch.setattr(broadcast_source, "_kept_sample_frames", lambda *a: list(range(1, 301)))
    result = broadcast_source.source_cadence(
        Path("source.mp4"), {"fps": 50.0, "duration_seconds": 60.0}
    )
    assert result["decision"] == "rejected"


def test_source_cadence_allows_millisecond_container_quantization(monkeypatch):
    monkeypatch.setattr(
        broadcast_source, "audit_source_timebase", lambda *a: {"nominal_timeline_valid": True}
    )
    fps = 30000 / 1001
    timestamps = [round(12.345 + index / fps, 3) for index in range(240)]
    monkeypatch.setattr(broadcast_source, "_sample_starts", lambda *a, **kw: [0.0])
    monkeypatch.setattr(broadcast_source, "_frame_timestamps", lambda *a: timestamps)
    monkeypatch.setattr(broadcast_source, "_kept_sample_frames", lambda *a: list(range(1, 241)))
    result = broadcast_source.source_cadence(
        Path("source.mkv"), {"fps": fps, "duration_seconds": 60.0}
    )
    assert result["decision"] == "usable"
    assert result["processing_fps"] == fps


def test_good_samples_cannot_override_a_failed_full_source_clock(monkeypatch):
    monkeypatch.setattr(broadcast_source, "_sample_starts", lambda *a, **kw: [0.0])
    monkeypatch.setattr(
        broadcast_source, "_frame_timestamps", lambda *a: [i / 25 for i in range(50)]
    )
    monkeypatch.setattr(broadcast_source, "_kept_sample_frames", lambda *a: list(range(1, 51)))
    monkeypatch.setattr(
        broadcast_source, "audit_source_timebase", lambda *a: {"nominal_timeline_valid": False}
    )
    result = broadcast_source.source_cadence(
        Path("source.mp4"), {"fps": 25.0, "duration_seconds": 600.0}
    )
    assert result["decision"] == "rejected"
    assert result["reason"] == "invalid_full_source_timebase"
    assert result["full_timebase"]["nominal_timeline_valid"] is False


def test_processing_source_never_replaces_a_missing_normalized_video(tmp_path):
    original = tmp_path / "original.mp4"
    original.touch()
    integrity = {
        "fps": 60.0,
        "cadence": {
            "decision": "healed",
            "processing_fps": 24.0,
            "normalized_video": "normalized.mp4",
        },
    }
    with pytest.raises(FileNotFoundError, match="normalized"):
        broadcast_source.processing_source(original, tmp_path, integrity)
    normalized = tmp_path / "normalized.mp4"
    normalized.touch()
    assert broadcast_source.processing_source(original, tmp_path, integrity) == (normalized, 24.0)


def test_native_processing_source_uses_original_path_not_a_same_named_output(tmp_path):
    original = tmp_path / "original.mp4"
    original.touch()
    out = tmp_path / "out"
    out.mkdir()
    (out / original.name).touch()
    integrity = {"fps": 25.0, "cadence": {"decision": "usable", "processing_fps": 25.0}}
    assert broadcast_source.processing_source(original, out, integrity) == (original, 25.0)
    integrity["cadence"]["processing_fps"] = 50.0
    with pytest.raises(ValueError, match="cadence"):
        broadcast_source.processing_source(original, out, integrity)


def _synthetic_board_crops(directory: Path, count: int = 24) -> list[Path]:
    """A wide crop whose left third holds a static bright board over churning background."""
    import cv2
    import numpy as np

    rng = np.random.default_rng(0)
    board = np.zeros((80, 200, 3), np.uint8)
    board[:] = (40, 120, 40)
    cv2.rectangle(board, (4, 4), (195, 75), (255, 255, 255), 2)
    cv2.putText(board, "SINNER", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(board, "ALCARAZ", (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    paths = []
    for index in range(count):
        frame = rng.integers(0, 255, (160, 600, 3), dtype=np.uint8)
        frame[30:110, 20:220] = board
        path = directory / f"s_{index:06d}.jpg"
        cv2.imwrite(str(path), frame)
        paths.append(path)
    return paths


def test_the_scoreboard_box_is_refined_inside_an_already_cut_crop(tmp_path) -> None:
    paths = _synthetic_board_crops(tmp_path)

    refined = broadcast_source.refine_scoreboard_box(paths, samples=24)

    assert refined is not None
    left, top, right, bottom = refined["box_normalised"]
    assert 0.0 <= left < 0.10 and right < 0.50
    assert top < 0.30 and bottom > 0.55
    assert refined["crop_size"] == [600, 160]


def test_tightening_re_cuts_and_enlarges_without_touching_the_video(tmp_path) -> None:
    import cv2

    paths = _synthetic_board_crops(tmp_path, count=4)
    out = tmp_path / "tight"

    written = broadcast_source.tighten_scoreboard_crops(
        paths, out, [0.0, 0.125, 0.4, 0.75], scale=2.0
    )

    assert written == 4
    image = cv2.imread(str(out / paths[0].name))
    assert image.shape[0] == 2 * (int(0.75 * 160) - int(0.125 * 160))
    assert image.shape[1] == 2 * int(0.4 * 600)


def _pair_conversion_samples(
    monkeypatch,
    phases: list[int],
    *,
    span: int = 400,
    source_fps: float = 50.0,
    ratio: tuple[int, int] = (1, 2),
) -> None:
    """25 fps samples converted to ``source_fps`` (default doubled to 50); ``phases`` gives
    each sample's conversion phase."""
    starts = [index * 100.0 for index in range(len(phases))]
    monkeypatch.setattr(
        broadcast_source, "audit_source_timebase", lambda *a: {"nominal_timeline_valid": True}
    )
    monkeypatch.setattr(broadcast_source, "_sample_starts", lambda *a, **kw: starts)
    monkeypatch.setattr(
        broadcast_source,
        "_frame_timestamps",
        lambda video, start, duration: [start + index / source_fps for index in range(span)],
    )
    by_start = dict(zip(starts, phases))

    def kept(video, start, duration, fps):
        first = round(start * fps)
        repeats = predicted_repeats(first, span, by_start[start], ratio)
        return [local for local in range(1, span + 1) if first + local - 1 not in repeats]

    monkeypatch.setattr(broadcast_source, "_kept_sample_frames", kept)


def test_restore_switch_restores_a_consistent_pair_conversion(monkeypatch) -> None:
    _pair_conversion_samples(monkeypatch, [1, 1, 1])
    metadata = {"fps": 50.0, "duration_seconds": 600.0}

    legacy = broadcast_source.source_cadence(Path("source.mp4"), metadata)
    restored = broadcast_source.source_cadence(Path("source.mp4"), metadata, restore=True)

    # Each sample opens mid-pair, so the concatenated legacy exposure pattern has lone frames
    # between samples and the content heal refuses a clean doubling.
    assert legacy["decision"] == "rejected"
    assert restored["decision"] == "restored"
    assert restored["processing_fps"] == 25.0
    assert restored["cadence_restore"]["phase"] == 1
    assert restored["cadence_restore"]["native_time_offset_seconds"] == pytest.approx(-0.02)
    assert [sample["phase"] for sample in restored["cadence_restore"]["samples"]] == [1, 1, 1]


def test_restore_switch_refuses_a_phase_that_changes_inside_the_broadcast(monkeypatch) -> None:
    _pair_conversion_samples(monkeypatch, [0, 1, 0])

    result = broadcast_source.source_cadence(
        Path("source.mp4"), {"fps": 50.0, "duration_seconds": 600.0}, restore=True
    )

    assert result["decision"] == "rejected"
    assert result["reason"] == "phase_changes_between_samples"


def test_restore_switch_leaves_three_two_conversions_to_the_legacy_action(monkeypatch) -> None:
    monkeypatch.setattr(
        broadcast_source, "audit_source_timebase", lambda *a: {"nominal_timeline_valid": True}
    )
    monkeypatch.setattr(broadcast_source, "_sample_starts", lambda *args, **kwargs: [0.0])
    monkeypatch.setattr(
        broadcast_source,
        "_frame_timestamps",
        lambda *args, **kwargs: [index / 60.0 for index in range(300)],
    )
    kept = []
    frame = 1
    for index in range(120):
        kept.append(frame)
        frame += 2 if index % 2 == 0 else 3
    monkeypatch.setattr(broadcast_source, "_kept_sample_frames", lambda *args, **kwargs: kept)

    result = broadcast_source.source_cadence(
        Path("source.mp4"), {"fps": 60.0, "duration_seconds": 60.0}, restore=True
    )

    assert result["decision"] == "healed"


def test_processing_source_uses_the_restored_stream(tmp_path):
    original = tmp_path / "original.mp4"
    original.touch()
    restored = tmp_path / "cadence_restored_source.mp4"
    restored.touch()
    integrity = {
        "fps": 50.0,
        "cadence": {
            "decision": "restored",
            "processing_fps": 25.0,
            "normalized_video": restored.name,
        },
    }
    assert broadcast_source.processing_source(original, tmp_path, integrity) == (restored, 25.0)


def test_restore_switch_sees_a_doubling_behind_a_still_picture(monkeypatch) -> None:
    _pair_conversion_samples(monkeypatch, [0, 0, 0])
    kept = broadcast_source._kept_sample_frames

    def with_still(video, start, duration, fps):
        frames = kept(video, start, duration, fps)
        # A two-second still (e.g. a graphic) in the first sample: one long equivalence group.
        return [frame for frame in frames if not 101 <= frame <= 200] if start == 0.0 else frames

    monkeypatch.setattr(broadcast_source, "_kept_sample_frames", with_still)
    metadata = {"fps": 50.0, "duration_seconds": 600.0}

    legacy = broadcast_source.source_cadence(Path("source.mp4"), metadata)
    restored = broadcast_source.source_cadence(Path("source.mp4"), metadata, restore=True)

    assert legacy["decision"] == "usable"
    assert restored["decision"] == "restored"
    assert restored["processing_fps"] == 25.0


def test_restore_switch_finds_a_pulldown_whose_stills_lower_the_unique_rate(monkeypatch) -> None:
    # A 25 -> 30 pulldown whose stills bring the unique rate under 24.75: 24 fps (4 in 5) is
    # then a candidate too, but only 25 fps (5 in 6) fits the repeats.
    _pair_conversion_samples(monkeypatch, [3, 3, 3], source_fps=30.0, ratio=(5, 6))
    kept = broadcast_source._kept_sample_frames

    def with_still(video, start, duration, fps):
        frames = kept(video, start, duration, fps)
        return [frame for frame in frames if not 11 <= frame <= 60] if start == 0.0 else frames

    monkeypatch.setattr(broadcast_source, "_kept_sample_frames", with_still)

    restored = broadcast_source.source_cadence(
        Path("source.mp4"), {"fps": 30.0, "duration_seconds": 600.0}, restore=True
    )

    assert restored["effective_unique_fps"] < 24.75
    assert restored["decision"] == "restored"
    assert restored["processing_fps"] == 25.0
    assert restored["cadence_restore"]["phase"] == 3
    tried = {
        row["native_fps"]: row["action"] for row in restored["cadence_restore"]["native_fps_tried"]
    }
    assert tried == {24.0: "refuse", 25.0: "restore_native_cadence"}
