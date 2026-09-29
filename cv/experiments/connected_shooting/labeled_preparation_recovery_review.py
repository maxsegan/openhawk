"""Compose explicitly reviewed no-fit recoveries while retaining all 89 attempts.

The selection manifest binds scored reports and written visual dispositions.
This is development measurement, with no independent airborne XYZ truth. Run
``--baseline REPORT --selection MANIFEST --output REPORT``; no fitting occurs.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from cv.pipeline import provenance
from scripts.shared_data import repository_root, resolve_shared_root

RUNG = "terminal_semantics_and_net_32px_x2_bounce2f_ray75cm_serve50cm"


def totals(rows: list[dict]) -> dict:
    return {
        "complete_points": sum(bool(r["after"]["complete_point"]) for r in rows),
        "attempt_denominator": len(rows),
        "accepted_flights": sum(int(r["after"]["accepted_flights"]) for r in rows),
        "labeled_flight_denominator": sum(int(r["labeled_flights"]) for r in rows),
        "points_with_at_least_one_accepted_flight": sum(
            int(r["after"]["accepted_flights"]) > 0 for r in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "selection", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root = resolve_shared_root(repository_root(), None)
    baseline = json.loads(args.baseline.read_text())
    selection = json.loads(args.selection.read_text())
    rows = deepcopy(baseline["attempts"])
    assert len(rows) == 89 and len({r["key"] for r in rows}) == 89
    by_key = {r["key"]: r for r in rows}
    assert len({r["key"] for r in selection["cases"]}) == len(selection["cases"])
    source_checks = []

    def bound(record: dict) -> Path:
        base = root if record["path_base"] == "TENNIS_DATA_ROOT" else repository_root()
        path = base / record["path"]
        actual = provenance.file_record(path)
        assert actual["sha256"] == record["sha256"], str(path)
        source_checks.append(actual)
        return path

    raw = []
    for case in selection["cases"]:
        row = by_key[case["key"]]
        assert row["baseline_document"] is None and row["after"]["accepted_flights"] == 0
        score = json.loads(bound(case["score"]).read_text())
        search = json.loads(bound(case["search"]).read_text())
        assert score["search_report"]["sha256"] == case["search"]["sha256"]
        assert score["reproduction_receipts"] and all(
            receipt["identical"] for receipt in score["reproduction_receipts"]
        )
        for record in score["inputs"]:
            bound(record)
        rung = next(r for r in score["rungs"] if r["rung"] == RUNG)
        verdict = rung["verdict"]
        assert verdict["flight_count"] == row["labeled_flights"]
        matches = [
            c
            for c in search["refined_candidates"]
            if abs(c["depth_hypothesis_m"] - rung["selected_depth_m"]) < 1e-10
        ]
        assert len(matches) == 1
        candidate = matches[0]
        summary = {
            "accepted_flights": verdict["accepted_flight_count"],
            "flight_count": verdict["flight_count"],
            "complete_point": verdict["complete_point"],
            "partial_point": verdict["partial_point"],
            "accepted_flight_indices": verdict["accepted_flight_indices"],
            "gaps": verdict["gaps"],
            "selected_depth_m": rung["selected_depth_m"],
        }
        raw.append(
            {
                **case,
                "summary": summary,
                "whole_point_rms_px": candidate["measurement"]["rms_px"],
                "player_contact_distances_m": candidate["evidence"]["player_distances_m"],
                "primary_optimizer": candidate["measurement"]["fit"].get("primary_optimizer"),
                "per_flight_reproduction_identical": True,
            }
        )
        if case["include_in_reviewed_composition"]:
            row["preparation_recovery_source"] = case
            row["before_preparation_recovery"] = deepcopy(row["after"])
            row["after"] = summary
    changed = [
        row["key"] for old, row in zip(baseline["attempts"], rows, strict=True) if old != row
    ]
    result = {
        "schema": "labeled_preparation_recovery_review_v1",
        "status": "complete",
        "human_derived": True,
        "automatic_inference_eligible": False,
        "independent_xyz_truth_available": False,
        "rung": RUNG,
        "summary": {"before": totals(baseline["attempts"]), "after": totals(rows)},
        "original_no_fit_denominator": 14,
        "measured_recovered_cases": len(raw),
        "raw_results": raw,
        "attempts": rows,
        "changed_attempts": changed,
        "unchanged_attempts": 89 - len(changed),
        "sources": [
            provenance.file_record(args.baseline),
            provenance.file_record(args.selection),
            provenance.file_record(Path(__file__)),
        ],
        "source_checks": source_checks,
        "limitations": selection["limitations"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
