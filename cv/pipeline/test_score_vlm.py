import json

from cv.pipeline.score_vlm import read_summary, write_stage_manifest


def test_score_read_summary_and_manifest_record_failed_reads(tmp_path) -> None:
    summary = read_summary(10, 7)
    write_stage_manifest(str(tmp_path), summary, status="failed_read_threshold")

    assert summary["read_failure_rate"] == 0.3
    document = json.loads((tmp_path / "score_vlm_manifest.json").read_text())
    assert document["status"] == "failed_read_threshold"
    assert document["reads_attempted"] == 10
    assert document["reads_ok"] == 7


def test_a_zero_points_box_beside_a_zero_games_box_is_not_a_duplicate() -> None:
    """Wimbledon at 4-6, 0-0, 15-0: the games box and the points box both read 0."""
    from cv.pipeline.score_vlm import norm_read

    read = norm_read(
        {
            "present": True,
            "rows": [
                {"name": "SINNER", "values": ["4", "0"], "points": "15"},
                {"name": "ALCARAZ", "values": ["6", "0"], "points": "0"},
            ],
            "serving_row": 2,
        }
    )

    assert read == {"g1": "0", "g2": "0", "p1": "15", "p2": "0", "sets1": "4", "sets2": "6"}


def test_a_tiebreak_number_repeated_into_the_values_is_still_dropped() -> None:
    from cv.pipeline.score_vlm import norm_read

    read = norm_read(
        {
            "present": True,
            "rows": [
                {"name": "SINNER", "values": ["6", "5"], "points": "5"},
                {"name": "ALCARAZ", "values": ["6", "3"], "points": "3"},
            ],
            "serving_row": 1,
        }
    )

    assert read == {"g1": "6", "g2": "6", "p1": "5", "p2": "3", "sets1": "", "sets2": ""}
