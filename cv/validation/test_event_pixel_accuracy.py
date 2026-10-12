from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from cv.validation.event_pixel_accuracy import (
    _native_columns,
    best_crop_candidate,
    canonical_fps,
    consensus_crop_recenter,
    load_track,
    owner_image_frame,
    percentile,
    select_event_anchor_pixel,
    summarize_fates,
)


def test_owner_half_frame_uses_the_clicked_following_image() -> None:
    assert owner_image_frame(10.0) == 10
    assert owner_image_frame(10.5) == 11


def test_native_columns_prefer_declared_native_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "track.csv"
    sidecar = path.with_name(f"{path.name}.coordinates.json")
    sidecar.write_text(
        json.dumps(
            {
                "artifact_size": {"width": 1920, "height": 1080},
                "coordinate_columns": {
                    "legacy_960x540": ["x", "y"],
                    "native_1920x1080": ["x_native", "y_native"],
                },
            }
        )
    )
    assert _native_columns(path, ["x", "y", "x_native", "y_native"]) == (
        "x_native",
        "y_native",
        1.0,
        1.0,
    )


def test_legacy_track_is_scaled_to_native(tmp_path: Path) -> None:
    path = tmp_path / "track.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip", "frame", "x", "y"])
        writer.writeheader()
        writer.writerow({"clip": "pt0001", "frame": "f_0003.jpg", "x": 100, "y": 50})
    path.with_name(f"{path.name}.coordinates.json").write_text(
        json.dumps(
            {
                "artifact_size": {"width": 960, "height": 540},
                "image_size": {"width": 1920, "height": 1080},
            }
        )
    )
    assert load_track(path) == {("pt0001", 3): (200.0, 100.0)}


def test_historical_candidate_without_sidecar_is_legacy_960x540(tmp_path: Path) -> None:
    path = tmp_path / "candidate.csv"
    assert _native_columns(path, ["x", "y"]) == ("x", "y", 2.0, 2.0)


def test_best_crop_candidate_uses_score_then_rank() -> None:
    rows = [
        {"x": 1, "y": 2, "score": 0.7, "rank": 1},
        {"x": 3, "y": 4, "score": 0.8, "rank": 2},
        {"x": 5, "y": 6, "score": 0.8, "rank": 0},
    ]
    assert best_crop_candidate(rows) == rows[2]
    assert best_crop_candidate([]) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [(24.0, "24"), (25.0, "25"), (50.0, "50"), (60000 / 1001, "59.94"), (60.0, "60")],
)
def test_canonical_fps(value: float, expected: str) -> None:
    assert canonical_fps(value) == expected


def test_percentile_ignores_missing_values() -> None:
    assert percentile([None, 1.0, 3.0], 50.0) == 2.0


def test_fate_summary_keeps_overlapping_rejections_explicit() -> None:
    rows = [
        {
            "scope": "test",
            "solved": True,
            "accepted": False,
            "reach_rejected": True,
            "anchor_violated": True,
            "head_contact_count": 1,
            "head_max_error_px": 30.0,
            "track_emitted_contact_count": 1,
            "track_emitted_max_error_px": 10.0,
            "crop_emitted_contact_count": 0,
            "crop_emitted_max_error_px": None,
        },
        {
            "scope": "test",
            "solved": True,
            "accepted": True,
            "reach_rejected": False,
            "anchor_violated": False,
            "head_contact_count": 1,
            "head_max_error_px": 3.0,
            "track_emitted_contact_count": 1,
            "track_emitted_max_error_px": 2.0,
            "crop_emitted_contact_count": 1,
            "crop_emitted_max_error_px": 4.0,
        },
    ]
    summary = summarize_fates(rows)
    indexed = {(row["fate"], row["source"]): row for row in summary}
    assert indexed[("reach_rejected", "head")]["flights"] == 1
    assert indexed[("anchor_violated", "head")]["flights"] == 1
    assert indexed[("reach_and_anchor", "head")]["flights"] == 1
    assert indexed[("accepted", "head")]["over_12px"] == 0


def test_event_anchor_uses_track_and_inflates_for_head_disagreement() -> None:
    result = select_event_anchor_pixel(
        event_type="contact", court_side="far", head=(40.0, 0.0), track=(10.0, 0.0)
    )
    assert (result["x"], result["y"]) == (10.0, 0.0)
    assert result["source"] == "automatic_track_at_emitted_frame"
    assert result["uncertainty_radius_px_p90"] == 30.0
    assert not result["abstain"]


def test_event_anchor_falls_back_to_head_and_can_abstain() -> None:
    fallback = select_event_anchor_pixel(
        event_type="bounce", court_side="near", head=(3.0, 4.0), track=None
    )
    assert (fallback["x"], fallback["y"]) == (3.0, 4.0)
    assert fallback["uncertainty_radius_px_p90"] == 30.0
    missing = select_event_anchor_pixel(
        event_type="bounce", court_side="near", head=None, track=None
    )
    assert missing["abstain"]


def test_crop_recenter_requires_agreement_with_head_and_track() -> None:
    point, adopted = consensus_crop_recenter(
        head=(10.0, 10.0), track=(11.0, 10.0), crop=(12.0, 10.0)
    )
    assert point == (12.0, 10.0)
    assert adopted
    point, adopted = consensus_crop_recenter(
        head=(10.0, 10.0), track=(11.0, 10.0), crop=(100.0, 100.0)
    )
    assert point == (11.0, 10.0)
    assert not adopted
