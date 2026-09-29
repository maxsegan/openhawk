import json
from pathlib import Path

import pytest

from cv.pipeline.cadence_normalize import (
    cadence_action,
    materialize_restored_frames,
    predicted_repeats,
    restore_plan,
    restore_ratio,
    restore_timeline,
)
from cv.pipeline.frame_cadence import audit_clip, classify_cadence


def test_empty_clip_is_an_explicit_timing_hold(tmp_path: Path) -> None:
    result = audit_clip(
        tmp_path,
        match_id="match",
        clip="pt0001",
        fps=25.0,
        active_spans=[],
    )

    assert result.decision == "timing_hold"
    assert result.frame_count == 0
    assert result.reasons == ["no_extracted_frames"]


def _regular_row(unique_fps: float, source_fps: float, exposures: int = 160) -> dict:
    ratio = source_fps / unique_fps
    lengths = [round((index + 1) * ratio) - round(index * ratio) for index in range(exposures)]
    groups = []
    duplicates = []
    frame = 1
    for length in lengths:
        if length > 1:
            group = list(range(frame, frame + length))
            groups.append(group)
            duplicates.extend(group[1:])
        frame += length
    result = classify_cadence(
        match_id="match",
        clip="pt0001",
        fps=source_fps,
        frame_count=frame - 1,
        kept_frames=sorted(set(range(1, frame)) - set(duplicates)),
        active_spans=[[1.0, float(frame - 1)]],
    ).as_dict()
    result["effective_unique_fps"] = unique_fps
    return result


@pytest.mark.parametrize(
    ("unique_fps", "source_fps"),
    [
        (24.0, 60.0),
        (24.0, 60000 / 1001),
        (25.0, 60.0),
        (30.0, 60.0),
        (25.0, 30.0),
        (24.0, 30.0),
        (24.0, 50.0),
        (25.0, 50.0),
        (24000 / 1001, 30000 / 1001),
    ],
)
def test_standard_upsample_patterns_hold_then_heal(unique_fps: float, source_fps: float) -> None:
    row = _regular_row(unique_fps, source_fps)

    assert row["decision"] == "timing_hold"
    action = cadence_action(row)
    assert action["action"] == "heal_regular_duplicate_cadence"
    assert action["target_fps"] == pytest.approx(unique_fps)


def test_irregular_duplicate_source_is_discarded() -> None:
    row = _regular_row(25.0, 30.0)
    groups = row["equivalence_groups"]
    # Preserve the same 1:2 hold counts but cluster them rather than following 4:5 cadence.
    groups[:] = groups[:1] + [group for group in groups[1:] if len(group) == 2]
    row["equivalence_groups"] = sorted(groups, key=lambda group: group[0])
    # Rebuild a deliberately front-loaded 1:2 source pattern.
    lengths = [2] * 40 + [1] * 120
    frame = 1
    groups = []
    duplicates = []
    for length in lengths:
        if length == 2:
            groups.append([frame, frame + 1])
            duplicates.append(frame + 1)
        frame += length
    row.update(
        frame_count=frame - 1,
        duplicate_frames=duplicates,
        duplicate_pair_starts=[frame - 1 for frame in duplicates],
        equivalence_groups=groups,
        effective_unique_fps=25.0,
    )

    assert cadence_action(row)["action"] == "discard_invalid_timebase"


def _converted_row(
    native_fps: float,
    source_fps: float,
    *,
    first: int = 0,
    count: int = 400,
    phase: int = 0,
    stills: tuple[int, ...] = (),
) -> dict:
    """A cadence row for absolute source frames [first, first + count) of a pair conversion."""
    ratio = restore_ratio(source_fps, native_fps)
    repeats = predicted_repeats(first, count, phase, ratio) | {first + s for s in stills}
    duplicates = sorted(frame - first + 1 for frame in repeats)
    return classify_cadence(
        match_id="match",
        clip="pt0001",
        fps=source_fps,
        frame_count=count,
        kept_frames=sorted(set(range(1, count + 1)) - set(duplicates)),
        active_spans=[[1.0, float(count)]],
    ).as_dict()


@pytest.mark.parametrize(
    ("native_fps", "source_fps", "phase"),
    [(25.0, 50.0, 0), (25.0, 50.0, 1), (25.0, 30.0, 0), (25.0, 30.0, 4), (30.0, 60.0, 1)],
)
def test_pair_conversions_restore_with_their_phase(native_fps, source_fps, phase) -> None:
    row = _converted_row(native_fps, source_fps, first=137, phase=phase)

    plan = restore_plan(row, first_source_frame=137)

    assert plan["action"] == "restore_native_cadence"
    assert plan["native_fps"] == native_fps
    assert plan["phase"] == phase
    timeline = restore_timeline(137, row["frame_count"], plan)
    captures = [item["native_capture"] for item in timeline]
    assert captures == list(range(captures[0], captures[0] + len(captures)))
    for item in timeline:
        # Every member of a native capture was shown after the capture instant, within a frame.
        for local in item["source_frames"]:
            shown = (137 + local - 1) / source_fps
            assert -1e-9 <= shown - item["native_time_seconds"] < 2.0 / source_fps + 1e-9
    assert all(len(item["source_frames"]) <= 2 for item in timeline)


def test_still_pictures_are_kept_as_distinct_native_captures() -> None:
    row = _converted_row(25.0, 50.0, stills=(42, 44, 46, 48))

    plan = restore_plan(row)

    assert plan["action"] == "restore_native_cadence"
    assert plan["unexplained_repeats"] == 4
    assert len(restore_timeline(0, 400, plan)) == 200


def test_three_two_holds_are_not_restored() -> None:
    row = _regular_row(24.0, 60.0)

    assert restore_plan(row)["reasons"] == ["not_an_isolated_pair_conversion"]


def test_a_supplied_wrong_phase_is_refuted() -> None:
    row = _converted_row(25.0, 50.0, phase=0)

    plan = restore_plan(row, phase=1)

    assert plan["action"] == "refuse"
    assert "predicted_repeats_not_observed" in plan["reasons"]


def test_irregular_pairs_with_a_pair_rate_are_refused() -> None:
    lengths = [2] * 40 + [1] * 120
    duplicates = []
    frame = 1
    for length in lengths:
        if length == 2:
            duplicates.append(frame + 1)
        frame += length
    row = classify_cadence(
        match_id="match",
        clip="pt0001",
        fps=30.0,
        frame_count=frame - 1,
        kept_frames=sorted(set(range(1, frame)) - set(duplicates)),
        active_spans=[[1.0, float(frame - 1)]],
    ).as_dict()
    row["effective_unique_fps"] = 25.0

    plan = restore_plan(row)

    assert plan["action"] == "refuse"
    assert "predicted_repeats_not_observed" in plan["reasons"]


def test_a_still_clip_cannot_choose_a_phase() -> None:
    row = _converted_row(25.0, 50.0, stills=tuple(range(1, 400)))

    plan = restore_plan(row, native_fps=25.0)

    assert plan["reasons"] == ["ambiguous_phase"]


def test_materialized_restore_keeps_one_picture_per_capture(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(1, 11):
        (source / f"f_{index:04d}.jpg").write_bytes(f"picture {(index + 1) // 2}".encode())
    row = _converted_row(25.0, 50.0, count=10, first=0, phase=0)
    plan = restore_plan(row, phase=0, native_fps=25.0)

    timeline = materialize_restored_frames(source, tmp_path / "native", first=0, plan=plan)

    written = sorted((tmp_path / "native").glob("f_*.jpg"))
    assert [path.read_bytes().decode() for path in written] == [
        f"picture {index}" for index in range(1, 6)
    ]
    assert [item["source_frames"] for item in timeline] == [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    mapping = json.loads((tmp_path / "native" / "cadence_restore_mapping.json").read_text())
    assert mapping["plan"]["phase"] == 0
