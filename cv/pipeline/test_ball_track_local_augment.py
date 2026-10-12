import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from ball_track_consensus import Candidate
from ball_track_local_augment import (
    _runs,
    apply_crop_authority,
    augment_track,
    candidate_family,
    fit_supported_runs,
    one_sided_extensions,
    recovery_config_for_fps,
    speed_safe_frames,
)


def test_runs_groups_only_consecutive_frames() -> None:
    assert _runs([2, 3, 4, 7, 9, 10]) == [(2, 4), (7, 7), (9, 10)]
    assert _runs([]) == []


def test_recovery_decoder_prefers_proposals_over_rank_metadata() -> None:
    config = recovery_config_for_fps(25)

    assert config.rank_cost < config.missing_cost / 50
    assert config.kink_cost < config.missing_cost


def test_one_sided_extension_uses_rank_inside_clean_projection() -> None:
    base = {frame: {"x": 100 + frame * 10, "y": 200 + frame * 5} for frame in range(1, 5)}
    candidates = {
        5: [
            Candidate(151, 226, 0.1, 8, ("motion_trajectory_forward",)),
            Candidate(158, 229, 0.2, 0, ("motion_trajectory_forward",)),
            Candidate(150, 225, 0.9, 0, ("motion_player_near",)),
        ]
    }

    recovered = one_sided_extensions(base, candidates, fps=25)

    assert recovered[5].x == 158
    assert "one_sided_extension" in recovered[5].sources


def test_one_sided_extension_rejects_jump_from_opposite_boundary() -> None:
    base = {
        **{frame: {"x": 800 - frame * 10, "y": 400 - frame * 5} for frame in range(6, 10)},
        2: {"x": 50, "y": 50},
    }
    candidates = {
        5: [
            Candidate(750, 375, 0.2, 0, ("motion_trajectory_backward",)),
        ]
    }

    recovered = one_sided_extensions(base, candidates, fps=25)

    assert recovered == {}


def test_speed_safe_frames_rejects_teleporting_fill_run() -> None:
    rows = {
        1: {"x": 0.0, "y": 0.0},
        2: {"x": 10.0, "y": 0.0},
        3: {"x": 200.0, "y": 0.0},
        4: {"x": 210.0, "y": 0.0},
    }
    assert speed_safe_frames(rows, {2, 3}, fps=25.0) == set()


def test_speed_safe_frames_retains_local_fill_run() -> None:
    rows = {
        1: {"x": 0.0, "y": 0.0},
        2: {"x": 10.0, "y": 0.0},
        3: {"x": 20.0, "y": 0.0},
        4: {"x": 30.0, "y": 0.0},
    }
    assert speed_safe_frames(rows, {2, 3}, fps=25.0) == {2, 3}


def test_contact_scale_context_accepts_supported_gap_run() -> None:
    points = {
        371: (729.375, 360.0),
        372: (742.5, 369.375),
        373: (755.625, 378.75),
        374: (768.75, 390.0),
        375: (783.75, 393.75),
        379: (699.375, 333.75),
        380: (679.75, 317.5),
        381: (650.37, 295.4),
        385: (545.625, 230.625),
        386: (525.0, 221.25),
        387: (506.25, 213.75),
        388: (487.5, 208.125),
        389: (474.375, 200.625),
    }
    rows = {
        frame: {
            "frame": f"f_{frame:04d}.jpg",
            "x": x,
            "y": y,
            "sources": (
                "local_gap_fill+alternate" if frame in {379, 380, 381} else "authoritative"
            ),
        }
        for frame, (x, y) in points.items()
    }

    assert fit_supported_runs(
        rows,
        [379, 380, 381],
        fps=25.0,
        context_seconds=0.2,
    ) == {379, 380, 381}


def test_candidate_family_reads_the_schema_not_the_name(tmp_path) -> None:
    crop = tmp_path / "ball_candidates_wasb_native1080_branched_crop_v2.csv"
    coarse = tmp_path / "ball_candidates_wasb_native1080_sliding_k5_v1.csv"
    far = tmp_path / "ball_candidates_wasb_native1080_far_native_v1.csv"

    assert candidate_family(crop, ["clip", "frame", "x", "y", "crop_provenance"]) == "crop"
    assert candidate_family(coarse, ["clip", "frame", "x", "y", "score"]) == "coarse"
    assert candidate_family(far, ["clip", "frame", "x", "y", "score"]) == "far_native"


def test_crop_detection_overwrites_the_quantised_coarse_lock() -> None:
    base = {
        frame: {"x": 100.0 + frame * 1.875, "y": 200.0, "sources": "sliding_wasb"}
        for frame in range(1, 4)
    }
    crop = {
        2: [Candidate(102.4, 200.9, 0.8, 0, ("crop_wasb", "crop_tracknetv2"))],
        3: [Candidate(400.0, 40.0, 0.9, 0, ("crop_wasb",))],
    }

    corrected, replaced = apply_crop_authority(base, crop, gate_px=24.0)

    assert replaced == 1
    assert (corrected[2]["x"], corrected[2]["y"]) == (102.4, 200.9)
    assert "crop_authoritative" in corrected[2]["sources"]
    # Frame 3's only crop candidate is far outside the gate: the coarse lock survives.
    assert corrected[3]["x"] == base[3]["x"]
    assert corrected[1] is base[1]


def test_crop_authority_prefers_the_detection_both_detectors_agree_on() -> None:
    base = {1: {"x": 100.0, "y": 100.0, "sources": "sliding_wasb"}}
    crop = {
        1: [
            Candidate(105.0, 100.0, 0.95, 0, ("crop_wasb",)),
            Candidate(101.0, 100.0, 0.4, 0, ("crop_wasb", "crop_tracknetv2")),
        ]
    }

    corrected, replaced = apply_crop_authority(base, crop, gate_px=24.0)

    assert replaced == 1
    assert corrected[1]["x"] == 101.0


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    lines = [",".join(header)]
    lines += [",".join(str(value) for value in row) for row in rows]
    path.write_text("\n".join(lines) + "\n")
    path.with_suffix(path.suffix + ".coordinates.json").write_text(
        json.dumps(
            {
                "schema": "tennis.coordinate-space.v1",
                "image_size": {"width": 1920, "height": 1080},
                "artifact_size": {"width": 1920, "height": 1080},
                "source": str(path),
            }
        )
    )


def test_augment_track_marks_provenance_per_row(tmp_path) -> None:
    base = tmp_path / "base.csv"
    _write_csv(
        base,
        ["clip", "frame", "x", "y", "track_id", "score", "rank", "sources"],
        [
            [
                "pt0001",
                f"f_{frame:04d}.jpg",
                100.0 + frame,
                200.0,
                0,
                0.9,
                0,
                "interpolated" if frame == 3 else "sliding_wasb",
            ]
            for frame in range(1, 6)
        ],
    )
    coarse = tmp_path / "ball_candidates_wasb_native1080_sliding_k5_v1.csv"
    _write_csv(
        coarse,
        ["clip", "frame", "x", "y", "score", "on_court", "rank"],
        [
            ["pt0001", f"f_{frame:04d}.jpg", 100.0 + frame, 200.0, 0.9, True, 0]
            for frame in range(1, 6)
        ],
    )
    crop = tmp_path / "ball_candidates_wasb_native1080_branched_crop_v2.csv"
    _write_csv(
        crop,
        ["clip", "frame", "x", "y", "score", "on_court", "rank", "crop_provenance"],
        [
            ["pt0001", f"f_{frame:04d}.jpg", 100.4 + frame, 200.3, 0.8, True, 0,
             "persistent_lock_observed"]
            for frame in (2, 3)
        ],
    )
    output = tmp_path / "augmented.csv"

    summary = augment_track(
        base,
        [coarse, crop],
        output,
        fps=25.0,
        crop_authoritative=True,
        crop_authority_px=24.0,
    )

    assert summary["crop_corrected_base_points"] == 2
    with output.open(newline="") as handle:
        rows = {int(row["frame"][2:6]): row for row in csv.DictReader(handle)}
    assert rows[1]["sources"] == "sliding_wasb+provenance:coarse"
    assert rows[2]["sources"].endswith("provenance:crop")
    assert float(rows[2]["x"]) == 102.4
    # A base row the coarse decoder interpolated is still reported as crop once a real
    # native detection replaces it.
    assert rows[3]["sources"].endswith("provenance:crop")


def test_augment_track_without_crop_authority_leaves_base_untouched(tmp_path) -> None:
    base = tmp_path / "base.csv"
    _write_csv(
        base,
        ["clip", "frame", "x", "y", "track_id", "score", "rank", "sources"],
        [["pt0001", f"f_{frame:04d}.jpg", 100.0 + frame, 200.0, 0, 0.9, 0, "sliding_wasb"]
         for frame in range(1, 6)],
    )
    crop = tmp_path / "ball_candidates_wasb_native1080_branched_crop_v2.csv"
    _write_csv(
        crop,
        ["clip", "frame", "x", "y", "score", "on_court", "rank", "crop_provenance"],
        [["pt0001", f"f_{frame:04d}.jpg", 100.4 + frame, 200.3, 0.8, True, 0,
          "persistent_lock_observed"] for frame in (2, 3)],
    )
    output = tmp_path / "augmented.csv"

    summary = augment_track(base, [crop], output, fps=25.0)

    assert summary["crop_corrected_base_points"] == 0
    with output.open(newline="") as handle:
        rows = {int(row["frame"][2:6]): row for row in csv.DictReader(handle)}
    assert float(rows[2]["x"]) == 102.0
    assert rows[2]["sources"] == "sliding_wasb"
