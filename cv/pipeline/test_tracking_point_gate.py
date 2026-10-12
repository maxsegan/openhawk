import csv
import json
from pathlib import Path

from cv.pipeline import resolution as res
from cv.pipeline.tracking_point_gate import (
    arc_point_decision,
    integer_scope_frames,
    maximum_gap_frames,
    motion_arcs,
    point_features,
    read_rows_by_clip,
    risk_weights,
    tracking_failure_reasons,
)


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    res.write_native_dual_coordinate_manifest(
        res.coordinate_manifest_path(path),
        image_size=res.NATIVE_SIZE,
        legacy_size=res.LEGACY_TRACKING_SIZE,
        source="test fixture",
        native_columns=("x_native", "y_native"),
        legacy_columns=("x", "y"),
    )


def test_scope_helpers_preserve_separate_span_boundaries():
    spans = [[2.2, 5.8], [10.0, 12.0]]
    assert integer_scope_frames(spans, 20) == {3, 4, 5, 10, 11, 12}
    assert maximum_gap_frames({3, 5, 10, 12}, spans, 20) == 1


def test_csv_rows_are_indexed_by_clip_once(tmp_path):
    path = tmp_path / "rows.csv"
    _write_csv(
        path,
        [
            {"clip": "pt0001", "frame": "f_0001.jpg"},
            {"clip": "pt0002", "frame": "f_0001.jpg"},
            {"clip": "pt0001", "frame": "f_0002.jpg"},
        ],
    )
    read_rows_by_clip.cache_clear()

    indexed = read_rows_by_clip(path)
    again = read_rows_by_clip(path)

    assert [row["frame"] for row in indexed["pt0001"]] == ["f_0001.jpg", "f_0002.jpg"]
    assert indexed is again


def test_failure_reasons_are_explicit_and_do_not_require_a_rank_budget():
    row = {
        "track_coverage_rate": 0.5,
        "maximum_track_gap_seconds": 0.2,
        "track_fragments_per_second": 0.4,
        "candidate_support_rate": 0.99,
        "nonballistic_window_rate": 0.1,
    }
    assert tracking_failure_reasons(row) == ["insufficient_live_track_coverage"]


def test_default_failure_thresholds_are_current_production():
    gates = json.loads(Path(__file__).with_name("default_gates.json").read_text())["gates"]
    thresholds = gates["tracking_failure_thresholds"]

    assert thresholds["minimum_track_coverage_rate"] == 0.6
    assert thresholds["minimum_candidate_support_rate"] == 0.94
    assert thresholds["maximum_nonballistic_window_rate"] == 0.45
    assert "tracking_failure_threshold_source" not in gates


def test_risk_profiles_are_explicit_and_keep_the_default_stable():
    gates = {
        "point_quality_risk_profile": "raw_fragment_count",
        "point_quality_risk_profiles": {
            "raw_fragment_count": {"track_fragments": 1.0},
            "fragment_rate_plus_gap": {
                "track_fragments_per_second": 1.0,
                "maximum_track_gap_seconds": 1.0,
            },
        },
    }
    assert risk_weights(gates) == ("raw_fragment_count", {"track_fragments": 1.0})
    assert risk_weights(gates, "fragment_rate_plus_gap") == (
        "fragment_rate_plus_gap",
        {
            "track_fragments_per_second": 1.0,
            "maximum_track_gap_seconds": 1.0,
        },
    )
    assert risk_weights({"point_quality_risk_weights": {"track_fragments": 1.0}}) == (
        "legacy_inline_weights",
        {"track_fragments": 1.0},
    )


def test_point_features_use_only_automatic_live_spans(tmp_path):
    clip = "pt0001"
    frame_dir = tmp_path / "audit_frames_native_1080" / clip
    frame_dir.mkdir(parents=True)
    for frame in range(1, 11):
        (frame_dir / f"f_{frame:04d}.jpg").touch()

    track_rows = [
        {
            "clip": clip,
            "frame": f"f_{frame:04d}.jpg",
            "x": frame,
            "y": frame,
            "score": 0.9,
            "sources": "sliding_tracknetv2+sliding_wasb",
        }
        for frame in range(1, 11)
    ]
    _write_csv(tmp_path / "ball_track_joint_native1080_arc_augmented_v2.csv", track_rows)
    candidate_names = (
        "ball_candidates_wasb_native1080_sliding_k5_v1.csv",
        "ball_candidates_tracknetv2_native1080_sliding_k5_v1.csv",
        "ball_candidates_wasb_native1080_branched_crop_v2.csv",
        "ball_candidates_tracknetv2_native1080_branched_crop_v2.csv",
    )
    candidate_rows = [
        {
            "clip": clip,
            "frame": f"f_{frame:04d}.jpg",
            "x": frame,
            "y": frame,
            "crop_provenance": "full_frame",
        }
        for frame in range(1, 11)
    ]
    for name in candidate_names:
        _write_csv(tmp_path / name, candidate_rows)

    result = point_features("match", tmp_path, clip, 25.0, spans=[[4, 7]])

    assert result["scope_source"] == "automatic_live_spans"
    assert result["scope_frames"] == 4
    assert result["track_points"] == 4
    assert result["track_coverage_rate"] == 1.0
    assert result["maximum_track_gap_seconds"] == 0.0
    assert result["whole_point_diagnostic_decision"] == "hold"


def test_motion_arcs_split_regimes_and_hold_failed_run():
    rows = []
    for frame in range(1, 9):
        regime = "ballistic" if frame <= 5 else "impulse"
        rows.append(
            {
                "clip": "pt0001",
                "frame": f"f_{frame:04d}.jpg",
                "x": float(frame),
                "y": float(frame),
                "regime": regime,
                "innovation_mahalanobis": 1.0,
                "innovation_cov_xx_native": 10.0,
                "innovation_cov_yy_native": 10.0,
            }
        )
    candidates = [
        [{"clip": row["clip"], "frame": row["frame"], "x": row["x"], "y": row["y"]} for row in rows]
    ]

    arcs = motion_arcs(rows, set(range(1, 9)), candidates)

    assert [arc["regime"] for arc in arcs] == ["ballistic", "impulse"]
    assert arcs[0]["decision"] == "retain"
    assert arcs[1]["decision"] == "hold"
    assert arcs[1]["failure_reasons"] == ["too_few_arc_observations"]
    assert arc_point_decision(arcs) == "retain"


def test_arc_point_decision_holds_only_when_no_arc_is_retained():
    assert arc_point_decision([]) == "unavailable"
    assert arc_point_decision([{"decision": "hold"}]) == "hold"
    assert arc_point_decision([{"decision": "hold"}, {"decision": "retain"}]) == "retain"
