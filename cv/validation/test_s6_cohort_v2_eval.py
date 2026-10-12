import json
from pathlib import Path

from cv.validation.s6_cohort_v2_eval import (
    build_mirror,
    percentile_pair,
    summarize_scope,
)


def test_build_mirror_filters_runner_event_view(tmp_path: Path) -> None:
    source = tmp_path / "source"
    match = source / "match"
    match.mkdir(parents=True)
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "matches": [
                    {"id": f"match_{index}", "point_ids": list(range(4))} for index in range(45)
                ]
                + [{"id": "match", "point_ids": list(range(18))}]
            }
        )
    )
    (match / "track.csv").write_text("frame,x,y\n")
    emissions = tmp_path / "emissions.json"
    rows = [
        {
            "abstain": False,
            "clip": "match__pt0001",
            "event_type": "contact",
            "frame": 1,
            "match_id": "match",
        },
        {
            "abstain": True,
            "clip": "match__pt0001",
            "event_type": "bounce",
            "frame": 2,
            "match_id": "match",
        },
        {
            "abstain": False,
            "clip": "match__pt0001",
            "event_type": "point_end",
            "frame": 3,
            "match_id": "match",
        },
    ]
    emissions.write_text(json.dumps(rows))
    emissions_manifest = tmp_path / "emissions.manifest.json"
    emissions_manifest.write_text(
        json.dumps(
            {
                "schema": "automatic_event_video_model_run_v1",
                "labels_or_reviewed_inputs": [],
            }
        )
    )

    output = tmp_path / "mirror"
    result = build_mirror(source, emissions, emissions_manifest, output)

    assert result["matches"] == 46
    assert result["points"] == 198
    assert json.loads((output / "event_emissions.json").read_text()) == [rows[0], rows[2]]
    assert (output / "match").is_symlink()
    manifest = json.loads((output / "event_emissions.manifest.json").read_text())
    assert manifest["selected_physical_rows"] == 1
    assert manifest["selected_point_end_rows"] == 1
    assert manifest["selected_abstained_rows"] == 0


def test_summarize_scope_splits_held_out_and_other() -> None:
    held_point = "ao2019f_w_osaka_kvitova__pt0001"
    other_point = "other_match__pt0001"
    fit = {
        "flight_index": 0,
        "held_out_reprojection_median_px": 2.0,
        "held_out_reprojection_p90_px": 4.0,
        "trajectory": [
            {"xyz": [5.0, 10.0, 2.0]},
            {"xyz": [5.0, 13.0, 2.0]},
        ],
    }
    report = {
        "points_detail": [
            {
                "point": held_point,
                "fits": [fit],
                "flight_attempts": [{"flight_index": 0}],
                "junction_gaps_m": [],
                "complete_point_gate": {"accepted": True},
                "reasons": [],
            },
            {
                "point": other_point,
                "fits": [],
                "flight_attempts": [{"flight_index": 0}],
                "junction_gaps_m": [],
                "complete_point_gate": {"accepted": False},
                "reasons": ["point_timeout"],
            },
        ]
    }
    ledger = {
        "rows": [
            {"point": held_point, "status": "provisional_valid"},
            {"point": other_point, "status": "unsolved"},
        ]
    }

    held = summarize_scope(
        report, ledger, {held_point, other_point}, "held_out_8", wall_time_seconds=3.0
    )
    other = summarize_scope(
        report, ledger, {held_point, other_point}, "other_38", wall_time_seconds=3.0
    )

    assert held["points"] == 1
    assert held["accepted_flights"] == 1
    assert held["accepted_complete_truth_points"] == 1
    assert held["net_crossings"] == 1
    assert held["wall_time_seconds"] is None
    assert other["points"] == 1
    assert other["solved_flights"] == 0
    assert other["point_timeouts"] == 1


def test_percentile_pair_handles_empty_and_values() -> None:
    assert percentile_pair([]) == [None, None]
    assert percentile_pair([1.0, 2.0, 3.0]) == [2.0, 2.8]
