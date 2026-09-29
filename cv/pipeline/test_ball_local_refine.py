import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import resolution as res
from ball_local_refine import (
    PlayerBox,
    RecoveryRegion,
    TrackPoint,
    build_persistent_lock_regions,
    build_recovery_regions,
    ensemble_target_heatmaps,
    heatmap_to_artifact,
    likely_players,
    refined_peaks,
    write_candidate_artifact,
    write_candidate_artifact_batches,
)


def test_likely_players_chooses_largest_box_per_half() -> None:
    boxes = [
        PlayerBox(10, 20, 30, 80, 0.9),
        PlayerBox(40, 30, 90, 100, 0.8),
        PlayerBox(100, 300, 180, 500, 0.9),
        PlayerBox(200, 320, 220, 380, 0.9),
    ]

    selected = dict(likely_players(boxes))

    assert selected["player_far"] == boxes[1]
    assert selected["player_near"] == boxes[2]


def test_build_regions_only_targets_bounded_missing_frames() -> None:
    track = [
        TrackPoint(1, 10, 20),
        TrackPoint(2, 20, 25),
        TrackPoint(5, 50, 40),
        TrackPoint(6, 60, 45),
    ]
    boxes = {
        ("pt0001", 3): [PlayerBox(100, 300, 180, 500, 0.9)],
        ("pt0001", 4): [PlayerBox(105, 300, 185, 500, 0.9)],
    }

    regions = build_recovery_regions("pt0001", 6, track, boxes, maximum_gap=3)

    assert {region.frame for region in regions} == {3, 4}
    assert {region.provenance for region in regions} == {
        "trajectory_forward",
        "trajectory_backward",
        "trajectory_bidirectional",
        "player_near",
    }


def test_heatmap_coordinates_map_to_artifact_crop() -> None:
    region = RecoveryRegion("pt0001", 10, 400, 250, 320, 180, "trajectory_forward")

    x, y = heatmap_to_artifact(255.5, 143.5, 512, 288, region)

    assert np.allclose((x, y), (400, 250))
    assert res.CANONICAL_SIZE == res.FrameSize(960, 540)


def test_crop_heatmap_peaks_default_to_centroid_refinement() -> None:
    ys, xs = np.mgrid[0:40, 0:40]
    heatmap = np.exp(-(((xs - 15.3) ** 2 + (ys - 22.7) ** 2) / (2 * 1.8**2))).astype(np.float32)

    (x, y, _), *_ = refined_peaks(heatmap, 1, 0.05, 3)

    assert np.hypot(x - 15.3, y - 22.7) < np.hypot(15 - 15.3, 23 - 22.7)


def test_crop_heatmap_argmax_remains_available() -> None:
    heatmap = np.zeros((20, 30), dtype=np.float32)
    heatmap[7, 11] = 1.0

    assert refined_peaks(heatmap, 1, 0.05, 3, "argmax")[0][:2] == (11.0, 7.0)


def test_persistent_lock_uses_true_native_crop_and_expands_through_gap() -> None:
    track = [
        TrackPoint(1, 100, 200),
        TrackPoint(2, 110, 200),
        TrackPoint(5, 140, 200),
        TrackPoint(6, 150, 200),
    ]

    regions = build_persistent_lock_regions(
        "pt0001",
        6,
        track,
        25.0,
        res.FrameSize(1920, 1080),
        res.CANONICAL_SIZE,
        maximum_extrapolation_seconds=0.4,
    )
    by_frame = {region.frame: region for region in regions}

    assert len(regions) == 6
    assert np.allclose((by_frame[1].width, by_frame[1].height), (256, 144))
    assert by_frame[3].provenance == "persistent_lock_bidirectional"
    assert np.allclose((by_frame[3].center_x, by_frame[3].center_y), (120, 200))
    assert by_frame[3].width > by_frame[1].width


def test_persistent_lock_branches_when_temporal_predictions_disagree() -> None:
    track = [
        TrackPoint(1, 100, 100),
        TrackPoint(2, 110, 100),
        TrackPoint(5, 300, 300),
        TrackPoint(6, 310, 300),
    ]

    regions = build_persistent_lock_regions(
        "pt0001",
        6,
        track,
        25.0,
        res.FrameSize(960, 540),
        res.CANONICAL_SIZE,
        maximum_extrapolation_seconds=0.4,
        branch_disagreement_px=45.0,
    )
    frame_three = [region for region in regions if region.frame == 3]

    assert len(frame_three) == 2
    assert {region.provenance for region in frame_three} == {
        "persistent_lock_forward_branch",
        "persistent_lock_backward_branch",
    }
    assert frame_three[0].center_x != frame_three[1].center_x


def test_ensemble_target_heatmaps_selects_directional_target_channels() -> None:
    outputs = np.zeros((1, 3, 3, 2, 2), dtype=float)
    outputs[:, 0, 2] = 3.0
    outputs[:, 1, 1] = 6.0
    outputs[:, 2, 0] = 9.0

    heatmaps = ensemble_target_heatmaps(outputs, (2, 1, 0))

    assert np.all(heatmaps == 6.0)


def test_candidate_writer_preserves_lock_coordinate_contract(tmp_path: Path) -> None:
    lock_track = tmp_path / "lock.csv"
    lock_track.write_text("clip,frame,x,y,x_native,y_native\n")
    res.write_native_dual_coordinate_manifest(
        res.coordinate_manifest_path(lock_track),
        image_size=res.NATIVE_SIZE,
        legacy_size=res.CANONICAL_SIZE,
        source="test",
        native_columns=("x_native", "y_native"),
        legacy_columns=("x", "y"),
    )
    output = tmp_path / "candidates.csv"

    write_candidate_artifact(
        output,
        [
            {
                "clip": "pt0001",
                "frame": "f_0001.jpg",
                "x": 100.0,
                "y": 200.0,
                "score": 0.9,
                "rank": 0,
            }
        ],
        lock_track,
    )

    with output.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    manifest = json.loads(res.coordinate_manifest_path(output).read_text())
    assert (float(row["x_native"]), float(row["y_native"])) == (200.0, 400.0)
    assert manifest["interface_space"] == "native_1920x1080"
    assert res.coordinate_manifest_errors(manifest) == []


def test_streaming_candidate_writer_is_byte_identical_to_buffered_writer(
    tmp_path: Path,
) -> None:
    lock_track = tmp_path / "lock.csv"
    lock_track.write_text("clip,frame,x,y,x_native,y_native\n")
    res.write_native_dual_coordinate_manifest(
        res.coordinate_manifest_path(lock_track),
        image_size=res.NATIVE_SIZE,
        legacy_size=res.CANONICAL_SIZE,
        source="test",
        native_columns=("x_native", "y_native"),
        legacy_columns=("x", "y"),
    )
    rows = [
        {
            "clip": "pt0001",
            "frame": f"f_{frame:04d}.jpg",
            "x": 100.0 + frame / 3,
            "y": 200.0 + frame / 7,
            "score": 0.9 - frame / 100,
            "on_court": True,
            "rank": frame - 1,
        }
        for frame in range(1, 4)
    ]
    buffered = tmp_path / "buffered" / "candidates.csv"
    streamed = tmp_path / "streamed" / "candidates.csv"

    write_candidate_artifact(buffered, rows, lock_track)
    count = write_candidate_artifact_batches(streamed, iter((rows[:1], rows[1:])), lock_track)

    assert count == len(rows)
    assert streamed.read_bytes() == buffered.read_bytes()
    assert (
        res.coordinate_manifest_path(streamed).read_bytes()
        == res.coordinate_manifest_path(buffered).read_bytes()
    )
